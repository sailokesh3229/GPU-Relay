"""Run a user script statement by statement and map the outcome to an exit code (spec §12.3).

Fresh run: every top-level statement runs live; a safe point follows each one.

Resume from a checkpoint taken in statement ``t``:

1. statements before ``t`` are **replayed** as setup (output suppressed, loops run fully)
   if they belong to the replay prefix or are definitions (imports/def/class); skipped
   statements that constructed torch objects are re-run too (after the snapshot's values are
   in place) when cheap, so optimizers etc. reference the live models; all other
   statements are **skipped** — their results come from the snapshot;
2. the snapshot's variables are put into the namespace;
3. statement ``t`` re-executes in replay mode up to the resume point, where state, RNG,
   and working files are restored and execution continues live.
"""

from __future__ import annotations

import builtins
import contextlib
import sys
import time
import traceback
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import gpurelay
from gpurelay.models import ExitCode
from gpurelay.runtime.errorloc import deepest, located
from gpurelay.runtime.instrument import (
    RELAY_NAME,
    InstrumentOptions,
    ScriptPlan,
    StatementInfo,
    plan_script,
)
from gpurelay.runtime.loops import ResumeMismatchError
from gpurelay.runtime.session import Handoff, RelayAbort, RelaySession

_PACKAGE_DIR = str(Path(gpurelay.__file__).resolve().parent)

OOM_HINT = (
    "[gpurelay] CUDA ran out of memory. Suggestions: reduce the batch size, enable mixed "
    "precision (torch.autocast / AMP), or free cached tensors. Not retried automatically."
)


@contextlib.contextmanager
def main_module(path: Path) -> Iterator[types.ModuleType]:
    """A fresh ``__main__`` module for ``path`` (like ``python path``)."""
    module = types.ModuleType("__main__")
    module.__dict__.update({"__file__": str(path), "__builtins__": builtins, "__spec__": None})
    saved_main = sys.modules.get("__main__")
    saved_argv = sys.argv[:]
    saved_path = sys.path[:]
    sys.modules["__main__"] = module
    sys.argv = [str(path)]
    sys.path.insert(0, str(path.parent))
    try:
        yield module
    finally:
        if saved_main is not None:
            sys.modules["__main__"] = saved_main
        sys.argv = saved_argv
        sys.path[:] = saved_path


def run_statements(
    plan: ScriptPlan,
    session: RelaySession,
    execute: Callable[[StatementInfo], None] | None = None,
) -> None:
    """Execute the planned statements (or notebook cells), replaying/skipping up to the
    resume point. ``execute`` runs one unit (default: ``exec`` of its compiled code)."""
    target = session.resume_target
    values_restored = False
    for stmt in plan.statements:
        if target is not None:
            t = int(target["stmt"])
            boundary = target["end"] == "stmt"
            if stmt.index < t or (stmt.index == t and boundary):
                if session.should_replay(stmt):
                    session.run_statement(stmt, passthrough=True, execute=execute)
                elif session.should_replay_constructor(stmt):
                    if not values_restored:
                        session.restore_values()  # inputs for the rebuilt objects
                        values_restored = True
                    session.run_statement(stmt, passthrough=True, execute=execute)
                else:
                    session.run_definitions(stmt)
                if stmt.index == t:
                    session.reach_boundary()
                    target = None
                continue
            if stmt.index == t:
                session.restore_values()
                session.run_statement(stmt, target=target, execute=execute)
                target = None
                session.statement_done(stmt, None)  # spans segments: never replayable
                continue
        started = time.perf_counter()
        session.run_statement(stmt, execute=execute)
        session.statement_done(stmt, time.perf_counter() - started)


def print_user_traceback(exc: BaseException) -> None:
    """Print a traceback without the runtime's own leading frames."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        tb = tb.tb_next
    traceback.print_exception(type(exc), exc, tb, file=sys.stderr)
    if "CUDA out of memory" in str(exc):
        print(OOM_HINT, file=sys.stderr)


def run_script(path: Path, session: RelaySession) -> tuple[ExitCode, str]:
    """Run the script under ``session``; return (exit code, status message)."""
    path = Path(path).resolve()
    try:
        plan = plan_script(path.read_bytes(), str(path), loop_file_id=path.name,
                           options=InstrumentOptions.from_settings(session.settings))  # fmt: skip
    except SyntaxError as exc:
        print_user_traceback(exc)
        return ExitCode.USER_ERROR, located(f"SyntaxError: {exc.msg}", exc.lineno)
    with main_module(path) as module:
        module.__dict__["__doc__"] = plan.docstring
        module.__dict__[RELAY_NAME] = session.relay_api
        session.attach_script(plan, module.__dict__)
        try:
            try:
                run_statements(plan, session)
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    return ExitCode.USER_ERROR, f"script called sys.exit({exc.code!r})"
            result = session.finish()
        except Handoff:
            return ExitCode.HANDOFF, "handed off at a safe point"
        except RelayAbort as exc:
            return exc.exit_code, str(exc)
        except ResumeMismatchError as exc:
            return ExitCode.NON_RESUMABLE, str(exc)
        except KeyboardInterrupt:
            return ExitCode.INFRA_ERROR, "interrupted"
        except Exception as exc:
            print_user_traceback(exc)
            script = str(path)
            where = deepest(exc.__traceback__, lambda f: True if f == script else None)
            return ExitCode.USER_ERROR, located(f"{type(exc).__name__}: {exc}",
                                                where[1] if where else None)  # fmt: skip
    if result is None or not result.ok:
        error = result.error if result is not None else None
        return ExitCode.INFRA_ERROR, f"final checkpoint failed: {error!r}"
    return ExitCode.COMPLETED, "completed"
