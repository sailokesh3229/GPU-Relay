"""GPU-budget ledger per backend (spec §13.2).

Budget models:

* ``weekly_rolling`` — at most ``budget_h`` GPU-hours in any 7-day window;
* ``monthly`` — ``budget_h`` per calendar month (UTC), resetting on the 1st;
* ``daily`` — ``budget_h`` per UTC day, resetting at midnight;
* ``unknown_opportunistic`` — no known budget (e.g. Colab): remaining is unknown;
* ``unlimited`` backends (local machine) have no budget at all.

Usage records come from finished segments; running segments are counted live. A user
correction (``gpurelay quota set``) pins remaining (or used) hours at a moment; usage after
it is subtracted until the window that contained it has passed.

Budgets can be given in another unit (plan P11): dollars (Modal) or credits (Lightning AI),
with a price per GPU hour. The ledger always counts measured GPU hours; the unit only
converts the allowance to hours and the remainder back for display.
"""

from __future__ import annotations

import calendar
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from gpurelay.controller.db import LedgerEntry, StateDB
from gpurelay.models import BudgetModel

WEEK_S = 7 * 24 * 3600.0
DAY_S = 24 * 3600.0
UNITS = ("h", "usd", "credits")


@dataclass(frozen=True)
class Budget:
    """How a backend's GPU time is limited."""

    model: BudgetModel
    hours: float | None  # None: unknown (opportunistic) or unlimited
    unlimited: bool = False
    estimate: bool = True  # the configured value is a default / guess (shown as "est.")
    unit: str = "h"  # "h", "usd" or "credits": what the platform grants
    cost_per_h: float = 1.0  # units per GPU hour (1.0 for "h")

    @classmethod
    def from_amount(
        cls, model: BudgetModel, amount: float, unit: str, cost_per_h: float, **kw: bool
    ) -> Budget:
        """A budget granted in ``unit`` (e.g. 30 usd a month at 0.59 usd per GPU hour)."""
        if unit not in UNITS or cost_per_h <= 0:
            raise ValueError(f"budget unit must be one of {UNITS} with a positive price")
        return cls(model, amount / cost_per_h, unit=unit, cost_per_h=cost_per_h, **kw)

    def amount(self, hours: float | None) -> float | None:
        """``hours`` of GPU time in this budget's unit."""
        return None if hours is None else hours * self.cost_per_h


@dataclass(frozen=True)
class LiveUsage:
    """A running segment's usage so far."""

    start_ts: float
    now_ts: float


@dataclass(frozen=True)
class QuotaState:
    budget: Budget
    used_h: float
    remaining_h: float | None  # None: unknown or unlimited
    next_reset: datetime | None
    corrected: bool = False

    @property
    def fraction_left(self) -> float | None:
        if self.remaining_h is None or not self.budget.hours:
            return None
        return self.remaining_h / self.budget.hours


def _overlap_h(start: float, end: float, lo: float, hi: float) -> float:
    return max(0.0, min(end, hi) - max(start, lo)) / 3600


def _day_bounds(now: float) -> tuple[float, float]:
    start = now - now % DAY_S
    return start, start + DAY_S


def _month_bounds(now: float) -> tuple[float, float]:
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    start = datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)
    days = calendar.monthrange(dt.year, dt.month)[1]
    return start.timestamp(), (start + timedelta(days=days)).timestamp()


class QuotaLedger:
    """Remaining budget and next reset per backend, from the ledger in :class:`StateDB`."""

    def __init__(self, db: StateDB) -> None:
        self.db = db

    @staticmethod
    def window(model: BudgetModel, now: float) -> tuple[float, float]:
        if model is BudgetModel.MONTHLY:
            return _month_bounds(now)
        if model is BudgetModel.DAILY:
            return _day_bounds(now)
        return now - WEEK_S, now

    def state(
        self,
        backend: str,
        budget: Budget,
        now: float,
        live: Iterable[LiveUsage] = (),
        *,
        min_segment_h: float = 0.0,
    ) -> QuotaState:
        entries = self.db.ledger(backend)
        intervals = [
            (e.start_ts, e.end_ts)
            for e in entries
            if e.kind == "usage" and e.start_ts is not None and e.end_ts is not None
        ] + [(u.start_ts, u.now_ts) for u in live]
        if budget.unlimited or budget.model is BudgetModel.UNKNOWN_OPPORTUNISTIC:
            lo, hi = now - WEEK_S, now
            used = sum(_overlap_h(s, e, lo, hi) for s, e in intervals)
            return QuotaState(budget, used, None, None)

        lo, hi = self.window(budget.model, now)
        used = sum(_overlap_h(s, e, lo, hi) for s, e in intervals)
        remaining = max(0.0, (budget.hours or 0.0) - used)
        corrected = False
        correction = self._latest_correction(entries, lo)
        if correction is not None:
            since = sum(_overlap_h(s, e, correction.created_ts, hi) for s, e in intervals)
            if correction.remaining_h is not None:
                remaining = max(0.0, correction.remaining_h - since)
            elif correction.used_h is not None:
                used = correction.used_h + since
                remaining = max(0.0, (budget.hours or 0.0) - used)
            corrected = True
        reset = self._next_reset(budget, now, intervals, remaining, min_segment_h)
        return QuotaState(budget, used, remaining, reset, corrected)

    @staticmethod
    def _latest_correction(entries: list[LedgerEntry], window_start: float) -> LedgerEntry | None:
        corrections = [
            e for e in entries if e.kind == "correction" and e.created_ts >= window_start
        ]
        return corrections[-1] if corrections else None

    @staticmethod
    def _next_reset(
        budget: Budget,
        now: float,
        intervals: list[tuple[float, float]],
        remaining: float,
        min_segment_h: float,
    ) -> datetime | None:
        if budget.model is BudgetModel.MONTHLY:
            return datetime.fromtimestamp(_month_bounds(now)[1], tz=timezone.utc)
        if budget.model is BudgetModel.DAILY:
            return datetime.fromtimestamp(_day_bounds(now)[1], tz=timezone.utc)
        # Rolling week: usage leaves the window 7 days after it happened. Find when enough
        # hours have expired to allow a minimum segment again.
        needed = max(min_segment_h, 1e-6) - remaining
        if needed <= 0:
            return None
        freed = 0.0
        for start, end in sorted(intervals, key=lambda iv: iv[1]):
            if end <= now - WEEK_S:
                continue
            freed += _overlap_h(start, end, now - WEEK_S, now)
            if freed >= needed:
                return datetime.fromtimestamp(end + WEEK_S, tz=timezone.utc)
        return datetime.fromtimestamp(now + WEEK_S, tz=timezone.utc)
