"""Scheduling policy (spec §13.4) — pure functions, no I/O.

1. Runnable jobs are ordered by priority (high first), then submission time.
2. A job goes to a backend that is ready, has remaining budget >= ``min_segment_h`` (or no
   budget limit), and has a free concurrency slot. Automated backends are preferred; among
   them the job's ``prefer_backend``, then the most remaining budget; when budgets tie, the
   GPU model of the job's previous segments (less numeric drift).
3. If only manual backends (Colab) can take it, the job waits for the user (``awaiting_user``).
4. If no backend has budget, the job waits for the earliest reset (``waiting_for_quota``).
5. Different jobs run in parallel on different backends (or up to ``max_concurrent`` on one);
   a job with a live segment is never scheduled again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

BUDGET_TIE_H = 0.5


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


@dataclass
class Decision:
    job_id: str
    action: str  # "launch" | "await_user" | "wait_quota" | "wait_slot"
    backend: str | None = None
    deadline_s: float | None = None
    wait_until: datetime | None = None
    reason: str = ""


def _remaining(backend: BackendView) -> float:
    return math.inf if backend.unlimited else (backend.remaining_h or 0.0)


def choose(candidates: list[BackendView], job: JobView) -> BackendView:
    """Preferred backend; else most remaining budget, ties (within 0.5 h) broken by the
    GPU model of the job's last segment, then by name for determinism."""
    for backend in candidates:
        if backend.name == job.prefer_backend:
            return backend
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
