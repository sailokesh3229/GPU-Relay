"""What GPU Relay asks the AI, and what it accepts back (spec §20).

The AI gets everything in the prompt (the code cell by cell, the error with its location,
the end of the log with the traceback) and answers with JSON: an explanation and the full new
source of each cell it changed. It never sees files or runs anything. The answer is checked
(cells that exist, real changes) and turned into before/after diffs for the job page.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_CODE_CHARS = 300_000  # larger jobs are not sent (the prompt would be too long)
LOG_TAIL_CHARS = 8_000  # the end of the log: the traceback and what led to it
MAX_EXPLANATION = 4_000

# The answer's shape (Claude Code --json-schema, Codex --output-schema: strict, so every
# property is required and no others are allowed).
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"cell": {"type": "integer"}, "source": {"type": "string"}},
                "required": ["cell", "source"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["explanation", "changes"],
    "additionalProperties": False,
}


class FixError(Exception):
    """The fix could not be made; the message says why, for the job page."""


def code_cells(source: bytes, entry: str) -> list[str]:
    """The job's code as the error location counts it: a notebook's code cells, or a script
    as one cell."""
    text = source.decode("utf-8", "replace")
    if not entry.endswith(".ipynb"):
        return [text]
    import nbformat

    return [str(c.source) for c in nbformat.reads(text, 4).cells if c.cell_type == "code"]


@dataclass
class FixRequest:
    job_id: str
    name: str
    entry: str
    cells: list[str]
    error: str  # the job's status message, location first ("cell 4, line 34: ...")
    cell: int | None = None  # where it failed: code cell counted from 1 (notebooks)
    line: int | None = None
    log_tail: str = ""
    platform: str = ""  # where it ran (kaggle, modal, colab, local)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def notebook(self) -> bool:
        return self.entry.endswith(".ipynb")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FixRequest:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)

    def check_size(self) -> None:
        if sum(len(c) for c in self.cells) > MAX_CODE_CHARS:
            raise FixError(f"the code is longer than {MAX_CODE_CHARS:,} characters, too long to "
                           "send to the AI in one piece")  # fmt: skip


def prompt(req: FixRequest) -> str:
    kind = "Jupyter notebook" if req.notebook else "Python script"
    where = (f"code cell {req.cell}, line {req.line}" if req.cell is not None
             else f"line {req.line}" if req.line is not None else "unknown")  # fmt: skip
    parts = [
        f"A {kind} ({req.entry}) failed while running on a cloud GPU"
        + (f" ({req.platform})" if req.platform else "")
        + ".",
        "Fix the error with the smallest change to the code. Keep the user's model, data, "
        "training and evaluation as they are; do not add features or restructure.",
        "If the error comes from the environment rather than the code (for example out of GPU "
        "memory), change only what makes it run (a smaller batch, a setting) and say so.",
        "If you cannot tell how to fix it, return no changes and explain what the user should "
        "check.",
        "",
        'Answer as JSON: "explanation" (two or three plain sentences: what was wrong and what '
        'you changed) and "changes" (for each code cell you changed, its number and its '
        "complete new source). Do not include cells you did not change.",
        "",
        f"Error ({where}):",
        req.error,
        "",
    ]
    if req.log_tail.strip():
        parts += ["End of the job's log (traceback):", req.log_tail.strip(), ""]
    for number, source in enumerate(req.cells, start=1):
        label = f"Code cell {number}" if req.notebook else "The script"
        parts += [f"----- {label} -----", source, ""]
    return "\n".join(parts)


def unified(before: str, after: str, label: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), f"{label} (before)", f"{label} (after)",
        lineterm="",
    )  # fmt: skip
    return "\n".join(lines)


def changes_from(req: FixRequest, answer: Any) -> tuple[str, list[dict[str, Any]]]:
    """(explanation, changes) from the AI's answer, checked; each change has the cell, its
    source before and after, and a unified diff."""
    if not isinstance(answer, dict):
        raise FixError("the AI's answer was not in the expected form")
    explanation = str(answer.get("explanation") or "").strip()[:MAX_EXPLANATION]
    raw = answer.get("changes")
    if not isinstance(raw, list):
        raise FixError("the AI's answer has no list of changes")
    changes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise FixError("the AI's answer has a change in an unexpected form")
        try:
            cell = int(item.get("cell", 0))
        except (TypeError, ValueError):
            raise FixError("the AI's answer names a cell that is not a number") from None
        if not req.notebook and cell == 0:
            cell = 1
        if not 1 <= cell <= len(req.cells):
            raise FixError(f"the AI changed cell {cell}, which does not exist")
        if cell in seen:
            raise FixError(f"the AI changed cell {cell} twice")
        seen.add(cell)
        before, after = req.cells[cell - 1], str(item.get("source", ""))
        if after.strip() == before.strip():
            continue  # "changed" without a change
        label = f"cell {cell}" if req.notebook else req.entry
        changes.append({"cell": cell, "before": before, "after": after,
                        "diff": unified(before, after, label)})  # fmt: skip
    changes.sort(key=lambda c: c["cell"])
    if not explanation and not changes:
        raise FixError("the AI gave neither a fix nor an explanation")
    return explanation, changes
