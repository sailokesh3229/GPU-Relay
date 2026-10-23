"""Human-friendly durations: ``90``, ``20s``, ``5m``, ``1.5h``, ``2d``."""

from __future__ import annotations

import re

_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)


def parse_duration(text: str, *, allow_zero: bool = False) -> float:
    """Return seconds. Raises ValueError for malformed or non-positive values (zero is
    accepted with ``allow_zero``, e.g. "no quota left")."""
    match = _RE.match(text)
    if not match:
        raise ValueError(f"invalid duration {text!r} (examples: 90, 20s, 5m, 1.5h)")
    seconds = float(match.group(1)) * _UNITS[match.group(2).lower()]
    if seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError(f"duration must be positive: {text!r}")
    return seconds


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"
