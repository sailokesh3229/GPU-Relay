"""Codex CLI on the user's computer, under their own ChatGPT plan (spec §20.2, plan A3).

GPU Relay runs the official ``codex`` command the user installed and signed in to (``codex
login``, "Sign in with ChatGPT"); it never reads their login. Each fix is one ``codex exec``
with the prompt on stdin, the answer checked against a JSON schema (``--output-schema``) and
written to a file (``-o``), in an empty temporary folder, not saved as a session
(``--ephemeral``).

Codex has no documented way to turn its tools off. It runs in its **read-only sandbox**: it
can run commands that read, never write; network is off by default (``docs/verify.md`` #32).
The code it needs is in the prompt, and the folder it starts in is empty. Claude Code, with
no tools at all, is therefore tried first when both are installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpurelay.aifix.request import ANSWER_SCHEMA, FixError, FixRequest
from gpurelay.aifix.tool import (
    RUN_TIMEOUT_S,
    Runner,
    ToolInfo,
    attempt,
    check_kwargs,
    last_line,
    quiet,
    tool_env,
)

NAME = "codex"
LABEL = "Codex"
DEFAULT_LABEL = "Codex's recommended model"


def detect(
    run: Runner = subprocess.run, which: Callable[[str], str | None] = shutil.which
) -> ToolInfo | None:
    """Codex if it is installed, with whether it is signed in (``codex login status`` exits 0
    when it has credentials)."""
    path = which("codex")
    if path is None:
        return None
    common = check_kwargs()
    try:
        version = run([path, "--version"], **common).stdout.strip().split(" ")[-1]
        status = run([path, "login", "status"], **common)
    except (OSError, subprocess.SubprocessError):
        return None
    said = f"{status.stdout or ''} {status.stderr or ''}".lower()
    plan = "chatgpt" if "chatgpt" in said else "api key" if "api key" in said else None
    return ToolInfo(NAME, LABEL, version or "?", status.returncode == 0, plan)


def command(path: str, model: str | None, schema: Path, answer: Path) -> list[str]:
    cmd = [
        path, "exec", "-", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
        "--output-schema", str(schema), "-o", str(answer),
    ]  # fmt: skip
    return cmd + (["-m", model] if model else [])


def ask(text: str, model: str | None, *, run: Runner = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
        timeout: float = RUN_TIMEOUT_S) -> tuple[Any, str]:  # fmt: skip
    """(the JSON answer, the model asked) for one prompt."""
    path = which("codex")
    if path is None:
        raise FixError("Codex is not installed on this computer")
    try:
        with tempfile.TemporaryDirectory(prefix="gpurelay-aifix-") as tmp:
            work, schema, answer = Path(tmp) / "work", Path(tmp) / "schema.json", Path(tmp) / "a"
            work.mkdir()
            schema.write_text(json.dumps(ANSWER_SCHEMA), encoding="utf-8")
            proc = run(command(path, model, schema, answer), input=text, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       cwd=work, env=tool_env(), **quiet())  # fmt: skip
            written = answer.read_text(encoding="utf-8") if answer.exists() else ""
    except subprocess.TimeoutExpired:
        raise FixError(f"Codex did not answer within {round(timeout / 60)} minutes") from None
    except OSError as exc:
        raise FixError(f"Codex could not be started: {exc}") from None
    if proc.returncode != 0 and not written.strip():
        raise FixError(f"Codex stopped: {last_line(proc.stderr)}")
    for text_answer in (written, proc.stdout or ""):
        try:
            return json.loads(text_answer), model or DEFAULT_LABEL
        except ValueError:
            continue
    raise FixError("Codex answered, but not in the expected form")


def fix(req: FixRequest, model: str | None = None, *,
        asker: Callable[[str, str | None], tuple[Any, str]] = ask,
        clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:  # fmt: skip
    return attempt(req, NAME, model, asker, clock)
