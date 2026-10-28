"""What every AI tool adapter shares (``claude_code``, ``codex``): what is reported about an
installed tool, how it is started, and how one call becomes a fix result."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from gpurelay.aifix.request import FixError, FixRequest, changes_from, prompt

RUN_TIMEOUT_S = 10 * 60.0
CHECK_TIMEOUT_S = 30.0
# GPU Relay's own tokens are never passed to an AI tool (it needs only its own login).
PRIVATE_ENV = ("GPURELAY_", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_", "MODAL_TOKEN")

Runner = Callable[..., Any]  # subprocess.run's signature
Asker = Callable[[str, "str | None"], tuple[Any, str]]  # (prompt, model) -> (answer, model)


@dataclass(frozen=True)
class ToolInfo:
    name: str
    label: str
    version: str
    signed_in: bool
    plan: str | None = None  # "pro", "max", "chatgpt", ... (never the account's email)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def quiet() -> dict[str, Any]:
    """No console window for the tool on Windows (the desktop app has none)."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def tool_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.upper().startswith(PRIVATE_ENV)}


def check_kwargs() -> dict[str, Any]:
    return dict(capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=CHECK_TIMEOUT_S, env=tool_env(), **quiet())  # fmt: skip


def last_line(text: str | None, default: str = "no answer") -> str:
    lines = (text or "").strip().splitlines()
    return (lines[-1] if lines else default)[:300]


def attempt(req: FixRequest, tool: str, model: str | None, asker: Asker,
            clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:  # fmt: skip
    """One tool and model asked for a fix: the result posted back, or why there is none."""
    started = clock()
    try:
        req.check_size()
        answer, used = asker(prompt(req), model)
        explanation, changes = changes_from(req, answer)
    except FixError as exc:
        return {"ok": False, "error": str(exc), "tool": tool, "model": model or "default",
                "seconds": round(clock() - started, 1)}  # fmt: skip
    return {"ok": True, "tool": tool, "model": used, "explanation": explanation,
            "changes": changes, "seconds": round(clock() - started, 1)}  # fmt: skip
