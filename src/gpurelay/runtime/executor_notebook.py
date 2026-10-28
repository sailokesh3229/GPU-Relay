"""Notebook executor (spec §7.1, Layer A).

Code cells run through IPython's ``InteractiveShell.run_cell`` in one shared ``__main__``
namespace, so magics, ``!cmd``, display hooks and the last-expression result behave as
in Jupyter. Each cell is a resumable unit, exactly like a script's top-level statement
(see ``executor_script``): short completed cells are replayed, long ones skipped and
restored from the snapshot, and an interrupted cell resumes inside its instrumented loop.

Per cell the executor records nbformat outputs (``stream``, ``display_data``,
``execute_result``, ``error``; matplotlib figures as ``image/png``). Outputs are stored as
content-addressed blobs referenced from every checkpoint (``runtime.cells``), so the final
``output.ipynb`` can be assembled with the outputs of cells that later segments skipped.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import nbformat

from gpurelay.models import ExitCode
from gpurelay.runtime.errorloc import deepest, located
from gpurelay.runtime.executor_script import main_module, run_statements
from gpurelay.runtime.instrument import (
    RELAY_NAME,
    CellInstrumenter,
    InstrumentOptions,
    StatementInfo,
    plan_cells,
)
from gpurelay.runtime.loops import ResumeMismatchError
from gpurelay.runtime.session import Handoff, RelayAbort, RelaySession

CELLS_KEY = "cells"


class CellError(Exception):
    """A notebook cell raised (its traceback is already in the cell's outputs)."""


class OutputRecorder:
    """nbformat outputs per code cell, persisted through the session's checkpoints."""

    def __init__(self, session: RelaySession) -> None:
        self.session = session
        meta = session.loaded.runtime if session.loaded else {}
        self.stored: dict[int, str] = {int(k): v for k, v in meta.get(CELLS_KEY, {}).items()}
        self.outputs: dict[int, list[dict[str, Any]]] = {}
        self.current: int | None = None

    # ------------------------------------------------------------------ recording

    def _recording(self) -> bool:
        return self.current is not None and self.session.capture.active

    def _append(self, output: dict[str, Any]) -> None:
        if self._recording():
            assert self.current is not None
            self.outputs.setdefault(self.current, []).append(output)

    def stream(self, name: str, text: str) -> None:
        if not self._recording():
            return
        assert self.current is not None
        outs = self.outputs.setdefault(self.current, [])
        if outs and outs[-1]["output_type"] == "stream" and outs[-1]["name"] == name:
            outs[-1]["text"] += text
        else:
            outs.append({"output_type": "stream", "name": name, "text": text})

    def display(self, data: dict[str, Any], metadata: dict[str, Any] | None) -> None:
        self._append(
            {"output_type": "display_data", "data": dict(data), "metadata": dict(metadata or {})}
        )

    def result(self, data: dict[str, Any], metadata: dict[str, Any] | None) -> None:
        self._append(
            {
                "output_type": "execute_result",
                "data": dict(data),
                "metadata": dict(metadata or {}),
                "execution_count": None,  # assigned at assembly
            }
        )

    def error(self, ename: str, evalue: str, traceback: list[str]) -> None:
        self._append(
            {"output_type": "error", "ename": ename, "evalue": evalue, "traceback": traceback}
        )

    def figure(self, png: bytes) -> None:
        self.display(
            {"image/png": base64.b64encode(png).decode("ascii"), "text/plain": "<Figure>"}, {}
        )

    # ------------------------------------------------------------------ cells

    def begin(self, index: int, live: bool) -> None:
        self.current = index if live else None
        if live:
            self.outputs[index] = []

    def end(self) -> None:
        self.current = None

    # ------------------------------------------------------------------ persistence

    def runtime_extra(self) -> tuple[dict[str, Any], set[str]]:
        cells = dict(self.stored)
        for index, outs in self.outputs.items():
            data = json.dumps(outs, sort_keys=True).encode("utf-8")
            cells[index] = self.session.blobs.put_bytes(data)
        return {CELLS_KEY: {str(k): v for k, v in sorted(cells.items())}}, set(cells.values())

    def restore_partial(self) -> None:
        """At the resume point: the interrupted cell keeps its outputs up to the checkpoint."""
        if self.current is not None and self.current in self.stored:
            self.outputs[self.current] = self.load(self.stored[self.current])

    def load(self, sha: str) -> list[dict[str, Any]]:
        outs: list[dict[str, Any]] = json.loads(self.session.blobs.read_bytes(sha))
        return outs


def _make_shell(module: Any, recorder: OutputRecorder, ipython_dir: Path) -> Any:
    os.environ.setdefault("IPYTHONDIR", str(ipython_dir))
    from IPython.core.displayhook import DisplayHook
    from IPython.core.displaypub import DisplayPublisher
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config import Config

    class RecordingDisplayHook(DisplayHook):  # type: ignore[misc]
        def write_output_prompt(self) -> None:
            pass

        def write_format_data(
            self, format_dict: dict[str, Any], md_dict: dict[str, Any] | None = None
        ) -> None:
            recorder.result(format_dict, md_dict)

    class RecordingDisplayPublisher(DisplayPublisher):  # type: ignore[misc]
        def publish(
            self,
            data: dict[str, Any],
            metadata: dict[str, Any] | None = None,
            source: Any = None,
            *,
            transient: Any = None,
            update: bool = False,
            **kwargs: Any,
        ) -> None:
            recorder.display(data, metadata)

        def clear_output(self, wait: bool = False) -> None:
            pass

    class RelayShell(InteractiveShell):  # type: ignore[misc]
        displayhook_class = RecordingDisplayHook
        display_pub_class = RecordingDisplayPublisher

        def _showtraceback(self, etype: Any, evalue: Any, stb: list[str]) -> None:
            recorder.error(getattr(etype, "__name__", str(etype)), str(evalue), stb)
            print(self.InteractiveTB.stb2text(stb), file=sys.stderr)

    config = Config()
    config.HistoryManager.enabled = False
    config.InteractiveShell.colors = "nocolor"
    return RelayShell.instance(config=config, user_module=module)


def job_python_first_on_path() -> None:
    """``!pip`` and ``!python`` in a cell run whatever comes first on PATH. On Kaggle and Colab
    that is the kernel's own Python; elsewhere (a virtual environment, a local run) it can be
    another installation, so packages would go to the wrong Python. Put this one first."""
    here = str(Path(sys.executable).parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if parts[:1] != [here]:
        os.environ["PATH"] = os.pathsep.join([here, *(p for p in parts if p != here)])


def run_notebook(path: Path, session: RelaySession) -> tuple[ExitCode, str]:
    """Run the notebook under ``session``; return (exit code, status message)."""
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    job_python_first_on_path()
    recorder = OutputRecorder(session)
    pending: list[BaseException] = []

    with main_module(path) as module:
        shell = _make_shell(module, recorder, session.runtime_dir / "ipython")
        try:
            options = InstrumentOptions.from_settings(session.settings)
            plan = plan_cells(sources, shell.transform_cell, name=path.name, options=options)
        except SyntaxError as exc:
            print(f"SyntaxError in notebook: {exc}", file=sys.stderr)
            return ExitCode.USER_ERROR, f"SyntaxError: {exc}"
        transformer = CellInstrumenter(plan)
        shell.ast_transformers.append(transformer)
        shell.set_custom_exc(
            (Handoff, RelayAbort, ResumeMismatchError, KeyboardInterrupt, SystemExit),
            lambda _shell, _etype, value, _tb, tb_offset=None: pending.append(value),
        )
        module.__dict__[RELAY_NAME] = session.relay_api
        session.attach_script(plan, module.__dict__)
        session.capture.stream_listener = recorder.stream
        session.capture.figure_listener = recorder.figure
        session.runtime_extras.append(recorder.runtime_extra)
        session.on_restore.append(recorder.restore_partial)

        runs: dict[int, int] = {}  # IPython execution number -> code cell index

        def cell_of(filename: str) -> int | None:
            number = shell.compile._filename_map.get(filename)
            return runs.get(number) if number is not None else None

        def execute(stmt: StatementInfo) -> None:
            assert stmt.source is not None
            recorder.begin(stmt.index, live=not session.tracker.passthrough)
            transformer.cell = stmt.index
            runs[shell.execution_count] = stmt.index
            try:
                result = shell.run_cell(stmt.source, store_history=True)
                if pending:
                    raise pending.pop()
                session.capture.capture_open_figures()  # like the inline backend
                if not result.success:
                    error = result.error_before_exec or result.error_in_exec
                    text = f"{type(error).__name__}: {error}"
                    # The failing line of the user's code (deepest frame in any cell).
                    where = deepest(error.__traceback__, cell_of)
                    cell, line = (where[0], where[1]) if where else (stmt.index, None)
                    if line is None and isinstance(error, SyntaxError) and error.lineno:
                        line = error.lineno
                    if line is None:
                        raise CellError(f"cell {cell + 1}: {text}")
                    raise CellError(located(text, line, cell + 1))
            finally:
                transformer.cell = None
                recorder.end()

        try:
            try:
                run_statements(plan, session, execute)
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    return ExitCode.USER_ERROR, f"notebook called sys.exit({exc.code!r})"
            result = session.finish()
        except Handoff:
            return ExitCode.HANDOFF, "handed off at a safe point"
        except RelayAbort as exc:
            return exc.exit_code, str(exc)
        except ResumeMismatchError as exc:
            return ExitCode.NON_RESUMABLE, str(exc)
        except KeyboardInterrupt:
            return ExitCode.INFRA_ERROR, "interrupted"
        except CellError as exc:
            return ExitCode.USER_ERROR, str(exc)
        finally:
            session.capture.stream_listener = None
            session.capture.figure_listener = None
    if result is None or not result.ok:
        error = result.error if result is not None else None
        return ExitCode.INFRA_ERROR, f"final checkpoint failed: {error!r}"
    return ExitCode.COMPLETED, "completed"


def build_output_notebook(
    source: nbformat.NotebookNode, cell_outputs: dict[int, list[dict[str, Any]]]
) -> nbformat.NotebookNode:
    """The source notebook with recorded outputs and sequential execution counts."""
    out = nbformat.from_dict(json.loads(json.dumps(source)))
    count = 0
    code_index = 0
    for cell in out.cells:
        if cell.cell_type != "code":
            continue
        outputs = cell_outputs.get(code_index)
        code_index += 1
        if outputs is None:
            cell.outputs = []
            cell.execution_count = None
            continue
        count += 1
        cell.execution_count = count
        cell.outputs = [nbformat.from_dict(o) for o in outputs]
        for output in cell.outputs:
            if output.output_type == "execute_result":
                output.execution_count = count
    nbformat.validate(out)
    return out


__all__ = ["CellError", "build_output_notebook", "run_notebook"]
