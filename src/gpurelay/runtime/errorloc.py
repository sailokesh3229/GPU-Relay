"""Where in the user's code an error happened, for the job page to point at.

The deepest frame of the traceback that lies in the user's own code is the line that
failed (inside a function, it is that function's line, not the call). The location goes at
the start of the status message, in a form people read and the web app parses:

* notebooks: ``cell 8, line 5: NameError: name 'train_loader' is not defined``
  (code cells counted from 1, lines counted within the cell);
* scripts: ``line 12: ZeroDivisionError: division by zero``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from types import TracebackType
from typing import TypeVar

T = TypeVar("T")

LOCATION_RE = re.compile(r"^(?:cell (?P<cell>\d+), )?line (?P<line>\d+): ")


def deepest(tb: TracebackType | None, owner: Callable[[str], T | None]) -> tuple[T, int] | None:
    """(owner, line) of the innermost frame whose file ``owner`` recognises."""
    found: tuple[T, int] | None = None
    while tb is not None:
        who = owner(tb.tb_frame.f_code.co_filename)
        if who is not None:
            found = (who, tb.tb_lineno)
        tb = tb.tb_next
    return found


def located(message: str, line: int | None, cell: int | None = None) -> str:
    """``message`` with its location in front (unchanged when the location is unknown)."""
    if line is None:
        return message
    where = f"cell {cell}, line {line}" if cell is not None else f"line {line}"
    return f"{where}: {message}"


def parse_location(message: str) -> tuple[int | None, int] | None:
    """(cell or None, line) from a status message made by :func:`located`."""
    match = LOCATION_RE.match(message or "")
    if match is None:
        return None
    cell = match.group("cell")
    return (int(cell) if cell else None, int(match.group("line")))
