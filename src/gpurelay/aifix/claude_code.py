"""Claude Code on the user's computer, under their own Claude plan (spec §20.2).

GPU Relay runs the official ``claude`` command the user installed and signed in to; it never
reads or handles their login. Each fix is one non-interactive call (``claude -p``) with **no
tools at all** (``--tools ""``): the code is in the prompt and the answer is JSON checked
against a schema, so the AI cannot read files, edit anything or run commands. It runs in an
empty temporary folder, without the user's Claude Code customizations (``--safe-mode``), MCP
servers or saved session.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
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

NAME = "claude_code"
LABEL = "Claude Code"
DEFAULT_MODEL = "sonnet"

__all__ = ["DEFAULT_MODEL", "LABEL", "NAME", "ToolInfo", "ask", "command", "detect", "fix"]


def detect(run: Runner = subprocess.run, which: Callable[[str], str | None] = shutil.which,
           ) -> ToolInfo | None:  # fmt: skip
    """Claude Code if it is installed, with whether it is signed in."""
    path = which("claude")
    if path is None:
        return None
    common = check_kwargs()
    try:
        version = run([path, "--version"], **common).stdout.strip().split(" ")[0]
        status = run([path, "auth", "status", "--json"], **common)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        auth = json.loads(status.stdout or "{}")
    except ValueError:
        auth = {}
    return ToolInfo(NAME, LABEL, version or "?", bool(auth.get("loggedIn")),
                    auth.get("subscriptionType") or None)  # fmt: skip


def command(path: str, model: str) -> list[str]:
    return [
        path, "-p", "--tools", "", "--model", model,
        "--output-format", "json", "--json-schema", json.dumps(ANSWER_SCHEMA),
        "--no-session-persistence", "--safe-mode", "--strict-mcp-config",
        "--permission-prompts", "none",
    ]  # fmt: skip


def ask(text: str, model: str | None, *, run: Runner = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
        timeout: float = RUN_TIMEOUT_S) -> tuple[Any, str]:  # fmt: skip
    """(the JSON answer, the model that gave it) for one prompt."""
    path = which("claude")
    if path is None:
        raise FixError("Claude Code is not installed on this computer")
    try:
        with tempfile.TemporaryDirectory(prefix="gpurelay-aifix-") as cwd:
            proc = run(command(path, model or DEFAULT_MODEL), input=text, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=cwd,
                       env=tool_env(), **quiet())  # fmt: skip
    except subprocess.TimeoutExpired:
        raise FixError(f"Claude Code did not answer within {round(timeout / 60)} minutes") from None
    except OSError as exc:
        raise FixError(f"Claude Code could not be started: {exc}") from None
    try:
        data = json.loads(proc.stdout or "")
    except ValueError:
        raise FixError(f"Claude Code stopped: {last_line(proc.stderr)}") from None
    if data.get("is_error"):
        reason = str(data.get("result") or data.get("subtype") or "unknown error")
        raise FixError(f"Claude Code: {reason[:300]}")
    answer = data.get("structured_output")
    if answer is None:
        try:
            answer = json.loads(str(data.get("result") or ""))
        except ValueError:
            raise FixError("Claude Code answered, but not in the expected form") from None
    used = next(iter(data.get("modelUsage") or {}), model or DEFAULT_MODEL)
    return answer, str(used)


def fix(req: FixRequest, model: str | None = DEFAULT_MODEL, *,
        asker: Callable[[str, str | None], tuple[Any, str]] = ask,
        clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:  # fmt: skip
    """The result posted back for a fix request: the explanation and changes, or why not."""
    return attempt(req, NAME, model or DEFAULT_MODEL, asker, clock)
