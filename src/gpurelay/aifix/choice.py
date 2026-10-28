"""Which AI to ask, in which order (spec §20.2, plan A3).

* **By the kind of error.** Simple faults (a misspelt name, a missing import, a wrong path)
  go to a fast model first; hard ones (tensor shapes, NaN loss, out of memory, anything not
  known as simple) go to the strongest first.
* **Escalation.** Each earlier fix of the same job that did not work (its rerun failed again,
  or the user asked again) moves the plan one step on: the next stronger model, then the
  other tool.
* **Learning.** When a rerun with a fix finishes, GPU Relay records whether that tool and
  model fixed that kind of error (``aifix/stats.json``). Once one has been tried a few times
  and fixes more often than the others, it is asked first.
* **Fallback.** A step whose tool is missing, or that fails (a usage limit, no answer), hands
  over to the next one in the same request.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpurelay.aifix import claude_code, codex
from gpurelay.aifix.request import FixRequest
from gpurelay.aifix.tool import ToolInfo
from gpurelay.config import AiFixSettings

SIMPLE = frozenset({
    "NameError", "UnboundLocalError", "ImportError", "ModuleNotFoundError", "SyntaxError",
    "IndentationError", "TabError", "AttributeError", "FileNotFoundError", "KeyError",
    "IndexError", "NotADirectoryError", "IsADirectoryError",
})  # fmt: skip
TYPE_RE = re.compile(r"(?:^|: )([A-Z][A-Za-z0-9_]*(?:Error|Exception|Interrupt))\b")
MIN_TRIES = 3  # before the record of a model counts
FIXERS: dict[str, Callable[[FixRequest, str | None], dict[str, Any]]] = {
    claude_code.NAME: claude_code.fix,
    codex.NAME: codex.fix,
}


def error_kind(error: str) -> str:
    """The exception's name (``cell 2, line 3: RuntimeError: ...`` -> ``RuntimeError``);
    running out of GPU memory counts as its own kind."""
    if "out of memory" in error.lower():
        return "OutOfMemoryError"
    match = TYPE_RE.search(error)
    return match.group(1) if match else "Error"


def detect_all() -> list[ToolInfo]:
    """The AI tools installed on this computer (Claude Code first: it runs with no tools)."""
    found = [claude_code.detect(), codex.detect()]
    return [t for t in found if t is not None]


class Stats:
    """Which tool and model fixed which kind of error: kind -> "tool/model" -> counts."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> dict[str, dict[str, dict[str, int]]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def record(self, kind: str, tool: str, model: str, fixed: bool) -> None:
        with self._lock:
            data = self.load()
            counts = data.setdefault(kind, {})
            entry = counts.setdefault(f"{tool}/{model}", {"tried": 0, "fixed": 0})
            entry["tried"] += 1
            entry["fixed"] += int(fixed)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)

    def rate(self, kind: str, tool: str, model: str) -> float | None:
        entry = self.load().get(kind, {}).get(f"{tool}/{model}")
        if not entry or entry.get("tried", 0) < MIN_TRIES:
            return None
        return float(entry["fixed"]) / float(entry["tried"])


def plan(req: FixRequest, settings: AiFixSettings, tools: list[dict[str, Any]],
         stats: Stats | None = None, earlier: int = 0) -> list[dict[str, Any]]:  # fmt: skip
    """The steps to try, in order: ``{"tool", "model"}`` (model None: the tool's default).

    ``tools``: the installed tools as reported (only signed-in ones are used; none known yet,
    as before a helper connects: Claude Code is assumed). ``earlier``: fixes of this job that
    did not work."""
    signed_in = {t.get("name") for t in tools if t.get("signed_in")}
    have = signed_in or {claude_code.NAME}
    kind = error_kind(req.error)
    fast: dict[str, Any] = {"tool": claude_code.NAME, "model": settings.claude_fast}
    strong: dict[str, Any] = {"tool": claude_code.NAME, "model": settings.claude_strong}
    other: dict[str, Any] = {"tool": codex.NAME, "model": settings.codex_model}
    order = [fast, strong, other] if kind in SIMPLE else [strong, other, fast]
    steps = [s for s in order if s["tool"] in have]
    if stats is not None:  # a model that has fixed this kind of error more often goes first
        rated = [(stats.rate(kind, s["tool"], str(s["model"] or "default")), i)
                 for i, s in enumerate(steps)]  # fmt: skip
        known = [(r, i) for r, i in rated if r is not None]
        if known:
            best_rate, best = max(known)
            if best_rate > 0 and all(best_rate > r for r, i in known if i != best):
                steps.insert(0, steps.pop(best))
    shift = min(earlier, max(0, len(steps) - 1))
    return steps[shift:] + steps[:shift]


def solve(req: FixRequest, steps: list[dict[str, Any]], available: set[str] | None = None,
          fixers: dict[str, Callable[[FixRequest, str | None], dict[str, Any]]] | None = None,
          ) -> dict[str, Any]:  # fmt: skip
    """Ask each step in turn until one gives a fix; the result says which were tried."""
    fixers = fixers or FIXERS
    tried: list[str] = []
    errors: list[str] = []
    for step in steps:
        tool, model = str(step.get("tool")), step.get("model")
        if tool not in fixers or (available is not None and tool not in available):
            continue
        result = fixers[tool](req, model)
        tried.append(f"{tool}/{result.get('model') or model or 'default'}")
        if result.get("ok"):  # "step": what the record of fixes counts it under
            return {**result, "tried": tried, "step": {"tool": tool, "model": model}}
        errors.append(str(result.get("error")))
    if not tried:
        return {"ok": False, "error": "no signed-in AI tool on this computer: install Claude "
                "Code (or Codex) and sign in", "tried": []}  # fmt: skip
    return {"ok": False, "error": "; ".join(dict.fromkeys(errors)), "tried": tried,
            "tool": tried[-1].split("/")[0], "model": tried[-1].split("/", 1)[1]}  # fmt: skip
