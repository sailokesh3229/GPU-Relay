"""Relative GPU speed for planning (plan P11): how much training work an hour on a GPU does,
compared with an hour on a Tesla T4 (= 1.0).

Rough, conservative figures for typical mixed-precision deep-learning training; real ratios
depend on the model and code. They only rank platforms and scale time estimates: budgets
and the ledger stay in measured GPU hours. Unknown GPUs count as a T4; "cpu" as 1/20.
"""

from __future__ import annotations

import re

# Longest names first so "a100" is not read as "a10".
SPEEDS: dict[str, float] = {
    "h100": 9.0,
    "a100": 5.0,
    "l40s": 4.0,
    "a10g": 2.2,
    "a10": 2.2,
    "v100": 2.4,
    "l4": 1.8,
    "p100": 1.3,
    "t4": 1.0,
    # Your own GPU (plan P15): consumer cards, by the same rough measure.
    "rtx4090": 4.0,
    "rtx4080": 3.0,
    "rtx4070": 2.0,
    "rtx4060": 1.4,
    "rtx3090": 2.6,
    "rtx3080": 2.2,
    "rtx3070": 1.6,
    "rtx3060": 1.2,
    "rtx3050": 0.8,
    "rtx2080": 1.4,
    "rtx2070": 1.1,
    "rtx2060": 0.9,
    "gtx1080": 0.8,
    "gtx1660": 0.6,
    "cpu": 0.05,
}
_NOISE = re.compile(r"nvidia|tesla|geforce|[\s_\-]+")
_MEMORY = re.compile(r"^\d+gb")


def normalize(label: str | None) -> str | None:
    """``"NvidiaTeslaT4"``, ``"Tesla T4"``, ``"T4"`` -> ``"t4"``; None if unknown."""
    if not label:
        return None
    key = _NOISE.sub("", label.lower())
    for name in sorted(SPEEDS, key=len, reverse=True):
        rest = key.removeprefix(name)
        if key == name or (key.startswith(name) and _suffix_only(rest)) or key.endswith(name):
            return name
    return None


def _suffix_only(rest: str) -> bool:
    """What follows a known name must not make it another model: ``"rtx3060laptopgpu"`` and
    ``"a10080gbpcie"`` (an A100 80GB) are fine; the ``"0"`` of ``"a100"`` is not an A10's."""
    return not rest or not rest[0].isdigit() or _MEMORY.match(rest) is not None


def speed_of(label: str | None, default: float = 1.0) -> float:
    """Relative speed of the GPU named ``label`` (T4 = 1.0)."""
    name = normalize(label)
    return SPEEDS[name] if name is not None else default
