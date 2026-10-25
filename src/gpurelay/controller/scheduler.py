"""Scheduling policy (spec §13.4) — pure functions, no I/O.

1. Runnable jobs are ordered by priority (high first), then submission time.
2. A job goes to a backend that is ready, has remaining budget >= ``min_segment_h`` (or no
   budget limit), and has a free concurrency slot. Automated backends are preferred; among
   them (plan P11):

   a. the job's ``prefer_backend``;
   b. budgets that expire (monthly, daily) while they still have hours, as long as their GPU
      is at least half as fast as the fastest candidate: unused credits are lost at reset,
      and among them the one resetting first;
   c. the fastest GPU (relative speed, T4 = 1.0): the job finishes soonest;
   d. the most remaining budget; when budgets tie, the GPU model of the job's previous
      segments (less numeric drift).
3. If only manual backends (Colab) can take it, the job waits for the user (``awaiting_user``):
   free GPU time always comes before paid.
4. Paid GPU time (plan P16), only when no free option is left and only within both the
   job's paid allowance and the platform's monthly paid limit (both $0 by default): the
   cheapest per unit of work (price / speed). The session ends before it could exceed
   either limit.
5. Otherwise the job waits for the earliest reset (``waiting_for_quota``).
5. Different jobs run in parallel on different backends (or up to ``max_concurrent`` on one);
   a job with a live segment is never scheduled again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

BUDGET_TIE_H = 0.5
EXPIRING_MIN_SPEED = 0.5  # expiring credits first, unless their GPU is under half as fast
SAME_SPEED = 0.95  # speeds within 5% count as equal
EXPIRING_MODELS = ("monthly", "daily")


@dataclass
class BackendView:
    name: str
    automated: bool
    ready: bool
    remaining_h: float | None  # None: unknown (manual/opportunistic) or unlimited
    unlimited: bool
    max_concurrent: int
    running: int
    max_session_s: float
    gpu_label: str | None = None
    next_reset: datetime | None = None
    budget_model: str | None = None  # BudgetModel value
    speed: float = 1.0  # relative GPU speed, T4 = 1.0
    paid_usd_per_h: float | None = None  # paid GPU time after the free budget
    paid_left_usd: float = 0.0  # the platform's monthly paid limit minus this month's spend

    @property
    def expiring(self) -> bool:
        """Hours that are lost if unused at the next reset (monthly / daily budgets)."""
        return (
            self.budget_model in EXPIRING_MODELS
            and not self.unlimited
            and (self.remaining_h or 0.0) > 0
        )

    def has_budget(self, min_segment_h: float) -> bool:
        if self.unlimited or self.remaining_h is None:
            return True
        return self.remaining_h >= min_segment_h

    def free(self) -> bool:
        return self.running < self.max_concurrent


@dataclass
class JobView:
    id: str
    priority: int
    submitted_at: datetime
    prefer_backend: str | None = None
    previous_gpus: list[str] = field(default_factory=list)
    paid_left_usd: float = 0.0  # the job's paid allowance minus what it already spent


@dataclass
class Decision:
    job_id: str
    action: str  # "launch" | "await_user" | "wait_quota" | "wait_slot"
    backend: str | None = None
    deadline_s: float | None = None
    wait_until: datetime | None = None
    reason: str = ""
    paid: bool = False  # the session runs on paid GPU time


def _remaining(backend: BackendView) -> float:
    return math.inf if backend.unlimited else (backend.remaining_h or 0.0)


def choose(candidates: list[BackendView], job: JobView) -> BackendView:
    """See the module docstring, rule 2: preferred backend; expiring credits on a GPU that
    is not much slower; the fastest GPU; the most remaining budget (ties within 0.5 h broken
    by the GPU model of the job's last segment, then by name for determinism)."""
    for backend in candidates:
        if backend.name == job.prefer_backend:
            return backend
    fastest = max(b.speed for b in candidates)
    expiring = [b for b in candidates if b.expiring and b.speed >= EXPIRING_MIN_SPEED * fastest]
    if expiring:
        soonest = min(b.next_reset.timestamp() if b.next_reset else math.inf for b in expiring)
        candidates = [
            b for b in expiring
            if (b.next_reset.timestamp() if b.next_reset else math.inf) <= soonest + 3600
        ]  # fmt: skip
    top = max(b.speed for b in candidates)
    candidates = [b for b in candidates if b.speed >= SAME_SPEED * top]
    best = max(_remaining(b) for b in candidates)
    tied = [b for b in candidates if _remaining(b) >= best - BUDGET_TIE_H]
    last_gpu = job.previous_gpus[-1] if job.previous_gpus else None
    return min(tied, key=lambda b: (b.gpu_label != last_gpu, -_remaining(b), b.name))


def deadline_for(backend: BackendView) -> float:
    """Session length: the backend's cap, never more than the remaining budget."""
    if backend.unlimited or backend.remaining_h is None:
        return backend.max_session_s
    return min(backend.max_session_s, backend.remaining_h * 3600)


def plan(
    jobs: list[JobView], backends: list[BackendView], *, min_segment_h: float
) -> list[Decision]:
    """Decide what each runnable job should do this tick (``jobs``: no live segment)."""
    slots = {b.name: b.running for b in backends}
    decisions: list[Decision] = []
    for job in sorted(jobs, key=lambda j: (-j.priority, j.submitted_at)):

        def usable(b: BackendView) -> bool:
            return b.ready and b.has_budget(min_segment_h) and slots[b.name] < b.max_concurrent

        automated = [b for b in backends if b.automated and usable(b)]
        if automated:
            chosen = choose(automated, job)
            slots[chosen.name] += 1
            decisions.append(
                Decision(job.id, "launch", chosen.name, deadline_for(chosen), reason="budget")
            )
            continue
        manual = [b for b in backends if not b.automated and usable(b)]
        if manual:
            chosen = manual[0]
            slots[chosen.name] += 1
            decisions.append(
                Decision(
                    job.id,
                    "await_user",
                    chosen.name,
                    deadline_for(chosen),
                    reason="only manual backends have budget",
                )
            )
            continue
        paid = [
            b for b in backends
            if b.automated and b.ready and b.paid_usd_per_h
            and slots[b.name] < b.max_concurrent
            and min(b.paid_left_usd, job.paid_left_usd) >= b.paid_usd_per_h * min_segment_h
        ]  # fmt: skip
        if paid:
            chosen = min(paid, key=lambda b: ((b.paid_usd_per_h or 0) / b.speed, b.name))
            slots[chosen.name] += 1
            allowed = min(chosen.paid_left_usd, job.paid_left_usd)
            deadline = min(chosen.max_session_s, allowed / (chosen.paid_usd_per_h or 1) * 3600)
            decisions.append(Decision(job.id, "launch", chosen.name, deadline,
                                      reason="paid, within your limits", paid=True))  # fmt: skip
            continue
        resets = [b.next_reset for b in backends if b.next_reset is not None and b.ready]
        busy = any(b.ready and b.has_budget(min_segment_h) for b in backends)
        decisions.append(
            Decision(
                job.id,
                "wait_quota" if not busy else "wait_slot",
                wait_until=min(resets) if resets and not busy else None,
                reason="all backends busy" if busy else "no backend has GPU budget left",
            )
        )
    return decisions
