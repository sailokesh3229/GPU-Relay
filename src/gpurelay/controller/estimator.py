"""Remaining GPU-time estimates (spec §13.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gpurelay.models import Job, StatusFile


@dataclass(frozen=True)
class Estimate:
    remaining_h: float | None
    basis: str  # "progress" | "hint" | "unknown"
    used_h: float


def estimate(job: Job, status: StatusFile | None, used_h: float) -> Estimate:
    """Remaining GPU-hours from the outermost loop's progress rate, else the user's hint."""
    if status is not None and status.rate is not None and status.rate.per_hour > 0:
        loops = list(status.progress.loop.values())
        if loops:
            completed, total = loops[0]
            if total is not None and total >= completed:
                return Estimate((total - completed) / status.rate.per_hour, "progress", used_h)
    if job.estimate_h is not None:
        return Estimate(max(0.0, job.estimate_h - used_h), "hint", used_h)
    return Estimate(None, "unknown", used_h)


@dataclass(frozen=True)
class BudgetLine:
    backend: str
    remaining_h: float | None  # None: unknown (manual/opportunistic)
    weekly_h: float | None  # budget per week (monthly budgets are scaled), None: unknown
    automated: bool
    unlimited: bool = False


def fit_message(needed_h: float | None, budgets: list[BudgetLine]) -> str:
    """Human summary (spec §13.3), e.g.
    'Job needs ~34.0 GPU-h. Fits this week: kaggle 22.0h + colab_manual (manual, ~unknown)'."""
    if needed_h is None:
        return "GPU time needed: unknown (pass --estimate, or wait for the first segment)"
    head = f"Job needs ~{needed_h:.1f} GPU-h."
    unlimited = [b.backend for b in budgets if b.unlimited and b.automated]
    if unlimited:
        return f"{head} Runs on {', '.join(unlimited)} (no quota)."
    left = needed_h
    parts: list[str] = []
    for line in sorted(budgets, key=lambda b: (not b.automated, -(b.remaining_h or 0.0))):
        if left <= 0 or line.remaining_h is None or line.remaining_h <= 0:
            continue
        take = min(line.remaining_h, left)
        parts.append(f"{line.backend} {take:.1f}h")
        left -= take
    # Backends with unknown budgets (manual Colab) are assumed to take the rest.
    unknown = [b for b in budgets if b.remaining_h is None and not b.unlimited]
    if left > 0 and unknown:
        line = unknown[0]
        kind = "manual" if not line.automated else "unknown budget"
        parts.append(f"{line.backend} ({kind}) ~{left:.1f}h")
        left = 0
    if left <= 0:
        return f"{head} Fits this week: {' + '.join(parts)}"
    weekly = sum(b.weekly_h for b in budgets if b.automated and b.weekly_h)
    if weekly <= 0:
        return f"{head} No automated backend has a known budget."
    weeks = math.ceil(needed_h / weekly)
    return (
        f"{head} Needs more than this week's budget: spans ~{weeks} week(s); will pause at "
        "budget exhaustion and resume after reset."
    )
