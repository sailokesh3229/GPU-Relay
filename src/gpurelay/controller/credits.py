"""Credit left on a paid platform (spec §21.2, plan G2).

The platform's own figure where its API reports one (fetched now and then: the balance minus
what GPU Relay's sessions cost since), otherwise the amount the user entered minus what GPU
Relay measured there since (price x time, :mod:`gpurelay.controller.spend`). The user can
correct the entered amount at any time; a correction restarts the count.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpurelay.controller.db import StateDB
from gpurelay.controller.spend import segment_cost
from gpurelay.models import Segment

LOW_FRACTION = 0.10  # "credit low" below 10 % of what was there
PLATFORM, ESTIMATED = "platform", "estimated"


@dataclass(frozen=True)
class CreditState:
    left_usd: float
    source: str  # PLATFORM (from the platform's API) or ESTIMATED (entered minus measured)
    as_of: float  # when the figure it starts from was known
    start_usd: float  # that figure


def cost_since(seg: Segment, since: float, now: float) -> float:
    """What ``seg`` cost after ``since`` (paid segments only)."""
    if not seg.paid_usd_per_h or seg.started_at is None:
        return 0.0
    start = max(seg.started_at.timestamp(), since)
    end = seg.ended_at.timestamp() if seg.ended_at else now
    return max(0.0, end - start) / 3600 * seg.paid_usd_per_h


def spent_since(db: StateDB, backend: str, since: float, now: float) -> float:
    """USD GPU Relay's paid sessions on ``backend`` cost after ``since``."""
    return sum(cost_since(s, since, now) for s in db.segments() if s.backend == backend)


def credit_state(
    db: StateDB,
    backend: str,
    now: float,
    *,
    entered_usd: float | None,
    entered_at: float | None,
    balance: tuple[float, float] | None = None,
) -> CreditState | None:
    """Credit left on ``backend``, or None when nobody knows (no balance, nothing entered).
    ``balance``: (USD, when) as the platform last reported it."""
    if balance is not None:
        value, at = balance
        left = value - spent_since(db, backend, at, now)
        return CreditState(max(0.0, left), PLATFORM, at, value)
    if entered_usd is None:
        return None
    since = entered_at if entered_at is not None else 0.0
    left = entered_usd - spent_since(db, backend, since, now)
    return CreditState(max(0.0, left), ESTIMATED, since, entered_usd)


def card_spent_this_month(db: StateDB, backend: str, now: float) -> float:
    """USD paid by card on ``backend`` this calendar month (credit sessions not counted)."""
    from gpurelay.controller.quota import _month_bounds

    start, end = _month_bounds(now)
    return sum(
        segment_cost(s, now)
        for s in db.segments()
        if s.backend == backend
        and s.paid_from != "credit"
        and s.started_at is not None
        and start <= s.started_at.timestamp() < end
    )
