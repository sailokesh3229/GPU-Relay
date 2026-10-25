"""Money spent on paid GPU time (plan P16).

A segment launched on paid time records its price (``Segment.paid_usd_per_h``); its cost is
that price times how long it ran. Free sessions cost nothing here, whatever the platform.
A platform's monthly spend counts segments started this calendar month (UTC).
"""

from __future__ import annotations

from gpurelay.controller.db import StateDB
from gpurelay.controller.quota import _month_bounds
from gpurelay.models import Segment


def segment_cost(seg: Segment, now: float) -> float:
    """USD this segment cost (or has cost so far, if still running); 0 when free."""
    if not seg.paid_usd_per_h or seg.started_at is None:
        return 0.0
    end = seg.ended_at.timestamp() if seg.ended_at else now
    return max(0.0, end - seg.started_at.timestamp()) / 3600 * seg.paid_usd_per_h


def job_spent(db: StateDB, job_id: str, now: float) -> float:
    """USD a job has spent on paid GPU time, in total."""
    return sum(segment_cost(s, now) for s in db.segments(job_id))


def backend_spent_this_month(db: StateDB, backend: str, now: float) -> float:
    """USD spent on ``backend``'s paid GPU time this calendar month (UTC)."""
    start, end = _month_bounds(now)
    return sum(
        segment_cost(s, now)
        for s in db.segments()
        if s.backend == backend
        and s.started_at is not None
        and start <= s.started_at.timestamp() < end
    )
