"""The controller loop (``gpurelay up``, spec §13).

Every tick:

1. apply user requests (pause / resume / cancel) recorded by the CLI;
2. poll live segments: finished, crashed (process gone, or heartbeat stale), or still running;
   record GPU usage in the ledger and move the job along its lifecycle;
3. assemble outputs of completed jobs;
4. schedule runnable jobs onto backends (``scheduler.plan``);
5. warn about low budgets.

``status.json`` (written by the runtime) is the source of truth for progress; backend state
is used to detect finished and crashed sessions.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from gpurelay.backends.common import session_settings
from gpurelay.backends.local import is_runtime_exit
from gpurelay.config import Config, RuntimeSettings
from gpurelay.constants import BUDGET_LOW_FRACTION
from gpurelay.controller.assemble import assemble
from gpurelay.controller.db import StateDB
from gpurelay.controller.lease import ControllerLease, LeaseLostError
from gpurelay.controller.notify import Notifier
from gpurelay.controller.quota import LiveUsage, QuotaLedger, QuotaState
from gpurelay.controller.registry import BackendSlot
from gpurelay.controller.scheduler import BackendView, JobView, deadline_for, plan
from gpurelay.errors import BackendBusyError
from gpurelay.models import (
    BackendRunState,
    ExitCode,
    Job,
    JobState,
    LaunchHandle,
    RuntimeState,
    Segment,
    SegmentState,
    StatusFile,
)
from gpurelay.runtime.checkpoint import read_pointer
from gpurelay.runtime.parallel import write_assignment
from gpurelay.storage.base import Storage
from gpurelay.storage.layout import control_key, status_key, worker_status_key

log = logging.getLogger(__name__)

SCHEDULABLE = (JobState.QUEUED, JobState.WAITING_FOR_QUOTA, JobState.AWAITING_USER)
STARTUP_GRACE_S = 900.0  # first heartbeat may take this long (boot + runtime install)
MAX_BACKOFF_FACTOR = 8  # failing ticks (storage/API outage) retry at up to 8x the interval
ALERT_AFTER_FAILURES = 5  # consecutive failing ticks before the user is notified
BUSY = (JobState.QUEUED, JobState.LAUNCHING, JobState.RUNNING, JobState.ASSEMBLING)

_OUTCOME_STATE = {
    "completed": SegmentState.COMPLETED,
    "handoff": SegmentState.HANDOFF,
    "crashed": SegmentState.CRASHED,
    "failed_user": SegmentState.FAILED_USER,
    "non_resumable": SegmentState.FAILED_USER,
    "failed_infra": SegmentState.FAILED_INFRA,
    "cancelled": SegmentState.CANCELLED,
}
_STATUS_OUTCOME = {
    RuntimeState.COMPLETED: "completed",
    RuntimeState.HANDOFF: "handoff",
    RuntimeState.FAILED_USER: "failed_user",
    RuntimeState.FAILED_INFRA: "failed_infra",
}


def _ts(value: datetime | None) -> float | None:
    return value.timestamp() if value is not None else None


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class Controller:
    """Scheduler + quota ledger + segment bookkeeping over a :class:`StateDB`."""

    def __init__(
        self,
        *,
        config: Config,
        db: StateDB,
        storage: Storage,
        backends: dict[str, BackendSlot],
        outputs_root: Path,
        notifier: Notifier | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.db = db
        self.storage = storage
        self.backends = backends
        self.outputs_root = Path(outputs_root)
        self.notifier = notifier or Notifier(config.notify)
        self.clock = clock
        self.quota = QuotaLedger(db)
        self._budget_warned: set[str] = set()
        # segment id -> (last heartbeat value seen, local time it was first seen). Staleness
        # is measured on this machine's clock only, so remote clock skew cannot fake it.
        self._beats: dict[str, tuple[datetime, float]] = {}
        # Cross-machine single-controller lease (set by `gpurelay up`; Phase 9 item 6).
        self.lease: ControllerLease | None = None

    # ------------------------------------------------------------------ loop

    def tick(self) -> None:
        now = self.clock()
        self._apply_requests(now)
        self._poll_segments(now)
        self._assemble_completed(now)
        self._schedule(now)
        self._launch_workers(now)
        self._check_budgets(now)

    def busy(self) -> bool:
        """Work left that the controller can make progress on by itself."""
        return bool(self.db.list_jobs(BUSY)) or bool(self.db.active_segments())

    def run(
        self,
        *,
        interval_s: float | None = None,
        until_idle: bool = False,
        max_ticks: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        interval = interval_s if interval_s is not None else self.config.scheduler.poll_interval_s
        ticks = failures = 0
        while True:
            if self.lease is not None:
                try:
                    self.lease.renew()
                except LeaseLostError:
                    raise  # another controller took over this store: stop
                except Exception as exc:
                    log.warning("could not renew the controller lease: %s", exc)
            try:
                self.tick()
                failures = 0
            except Exception as exc:  # storage or platform API outage: keep going, back off
                failures += 1
                log.error("controller tick failed (%s: %s); retrying", type(exc).__name__, exc)
                if failures == ALERT_AFTER_FAILURES:
                    self.notifier.notify(
                        "controller_error", "controller cannot make progress",
                        f"{failures} attempts failed in a row; last error: {exc}",
                    )  # fmt: skip
            ticks += 1
            if until_idle and not self.busy():
                return
            if max_ticks is not None and ticks >= max_ticks:
                return
            sleep(interval * min(2 ** max(failures - 1, 0), MAX_BACKOFF_FACTOR))

    # ------------------------------------------------------------------ helpers

    def _event(self, kind: str, message: str, now: float, job: Job | None = None,
               segment: Segment | None = None) -> None:  # fmt: skip
        self.db.add_event(
            kind,
            message,
            now=now,
            job_id=job.id if job else None,
            segment_id=segment.id if segment else None,
        )
        log.info("[%s] %s%s", kind, f"{job.id}: " if job else "", message)

    def _set_job(
        self, job: Job, state: JobState, now: float, message: str = "", **fields: object
    ) -> Job:
        job = job.model_copy(update={"state": state, "message": message, **fields})
        self.db.put_job(job)
        self._event(f"job_{state.value}", message or state.value, now, job)
        return job

    def _read_status(self, job_id: str, segment: Segment | None = None) -> StatusFile | None:
        """The job's ``status.json`` (main session), or a worker segment's heartbeat."""
        if segment is not None and segment.role == "worker":
            raw = self.storage.get_bytes(worker_status_key(job_id, segment.id))
        else:
            raw = self.storage.get_bytes(status_key(job_id))
        try:
            return StatusFile.model_validate_json(raw) if raw else None
        except ValueError:
            return None

    def _handle(self, segment: Segment) -> LaunchHandle:
        return LaunchHandle(
            backend=segment.backend,
            handle_id=segment.handle_id or "",
            extra=dict(segment.handle_extra),
        )

    def _live_usage(self, backend: str, now: float) -> list[LiveUsage]:
        return [
            LiveUsage(seg.started_at.timestamp(), now)
            for seg in self.db.active_segments()
            if seg.backend == backend and seg.started_at is not None
        ]

    def quota_state(self, slot: BackendSlot, now: float) -> QuotaState:
        return self.quota.state(
            slot.name,
            slot.budget,
            now,
            self._live_usage(slot.name, now),
            min_segment_h=self.config.scheduler.min_segment_h,
        )

    # ------------------------------------------------------------------ 1. requests

    def _apply_requests(self, now: float) -> None:
        for job in self.db.list_jobs():
            if job.requested is None:
                continue
            active = [s for s in self.db.segments(job.id) if s.state in _ACTIVE]
            if job.requested == "cancel":
                for seg in active:
                    self._stop_segment(seg, now, "cancelled")
                self._set_job(job, JobState.CANCELLED, now, "cancelled by user", requested=None)
            elif job.requested == "pause":
                running = [s for s in active if s.state is SegmentState.RUNNING]
                for seg in active:
                    if seg.state is not SegmentState.RUNNING:
                        self._stop_segment(seg, now, "cancelled")
                if running:
                    control = {"handoff": True, "segment": running[0].id, "reason": "pause"}
                    # The segment hands off at its next safe point; then the job is PAUSED.
                    self.storage.put_bytes(json.dumps(control).encode(), control_key(job.id))
                elif not job.state.is_terminal:
                    self._set_job(job, JobState.PAUSED, now, "paused by user", requested=None)
            elif job.requested == "resume":
                if job.state is JobState.PAUSED:
                    self._set_job(job, JobState.QUEUED, now, "resumed by user", requested=None)
                else:
                    self.db.put_job(job.model_copy(update={"requested": None}))

    def _stop_segment(self, seg: Segment, now: float, outcome: str) -> None:
        slot = self.backends.get(seg.backend)
        if slot is not None and seg.handle_id:
            try:
                slot.backend.cancel(self._handle(seg))
            except Exception as exc:
                log.warning("cancel of %s failed: %s", seg.id, exc)
        self._close_segment(seg, outcome, None, now, "stopped by the controller")

    # ------------------------------------------------------------------ 2. segments

    def _poll_segments(self, now: float) -> None:
        for seg in self.db.active_segments():
            try:
                self._poll_segment(seg, now)
            except Exception as exc:  # one broken segment must not stall the others
                log.warning("polling %s failed (%s: %s)", seg.id, type(exc).__name__, exc)

    def _poll_segment(self, seg: Segment, now: float) -> None:
        slot = self.backends.get(seg.backend)
        if slot is None:
            self._finish(seg, "crashed", None, now, "backend no longer configured")
            return
        if seg.state is SegmentState.AWAITING_USER:
            self._poll_manual(seg, slot, now)
            return
        handle = self._handle(seg)
        state = slot.backend.poll(handle)
        status = self._read_status(seg.job_id, seg)
        if status is not None and status.segment_id != seg.id:
            status = None
        if state is BackendRunState.QUEUED:
            return  # waiting in the platform's queue: no heartbeat expected yet
        if state is BackendRunState.RUNNING:
            if self._stale(seg, status, now, factor=2.0):
                slot.backend.cancel(handle)
                self._finish(seg, "crashed", None, now, "heartbeat stale; session killed")
            return
        if state is BackendRunState.UNKNOWN:
            if status is not None and status.state is not RuntimeState.RUNNING:
                self._finish(seg, _STATUS_OUTCOME[status.state], None, now, status.message,
                             ended=self._session_end(seg, status, now))  # fmt: skip
            elif self._stale(seg, status, now):
                self._finish(seg, "crashed", None, now, "lost contact (stale heartbeat)",
                             ended=self._session_end(seg, status, now))  # fmt: skip
            return
        code = slot.backend.returncode(handle) if hasattr(slot.backend, "returncode") else None
        outcome = self._classify(code, status)
        if seg.role == "worker" and code == ExitCode.COMPLETED:
            outcome = "completed"  # also "nothing to do" (exits before any status is written)
        self._finish(seg, outcome, code, now, status.message if status else "",
                     ended=self._session_end(seg, status, now))  # fmt: skip

    def _poll_manual(self, seg: Segment, slot: BackendSlot, now: float) -> None:
        """Manual (Colab) sessions: the runtime's status.json shows the user started it."""
        status = self._read_status(seg.job_id)
        if status is not None and status.segment_id == seg.id:
            # GPU time (and the session's deadline) start now, not at the request.
            update: dict[str, Any] = {"state": SegmentState.RUNNING, "started_at": _dt(now)}
            if seg.started_at is not None and seg.deadline_at is not None:
                update["deadline_at"] = _dt(now) + (seg.deadline_at - seg.started_at)
            seg = seg.model_copy(update=update)
            self.db.put_segment(seg)
            job = self.db.get_job(seg.job_id)
            if job is not None:
                self._set_job(job, JobState.RUNNING, now, "user started the session")

    def _runtime_for(self, seg: Segment) -> RuntimeSettings:
        """The settings the segment's session runs with (Hub-backed ones beat less often)."""
        slot = self.backends.get(seg.backend)
        if slot is not None and getattr(slot.backend, "hub_sessions", False):
            return session_settings(self.config.runtime)
        return self.config.runtime

    def _session_end(self, seg: Segment, status: StatusFile | None, now: float) -> float:
        """When a session that is no longer running ended, for its end time and GPU usage.

        Its last heartbeat (plus one interval), not this poll: the controller may have been
        down for hours while the session had long finished.
        """
        if status is None:
            return now
        last = status.heartbeat_at.timestamp() + self._runtime_for(seg).heartbeat_s
        return min(now, max(_ts(seg.started_at) or now, last))

    def _stale(
        self, seg: Segment, status: StatusFile | None, now: float, factor: float = 1.0
    ) -> bool:
        limit = self._runtime_for(seg).stale_heartbeat_s * factor
        if status is None:
            # No heartbeat yet: the session may still be booting / installing the runtime.
            started = _ts(seg.started_at) or now
            return now - started > limit + STARTUP_GRACE_S
        # Compare heartbeats with each other, not with this machine's clock: a session whose
        # clock is off by minutes is still alive as long as its heartbeat keeps changing.
        seen = self._beats.get(seg.id)
        if seen is None or seen[0] != status.heartbeat_at:
            self._beats[seg.id] = (status.heartbeat_at, now)
            return False
        return now - seen[1] > limit

    @staticmethod
    def _classify(code: int | None, status: StatusFile | None) -> str:
        if code is not None and is_runtime_exit(code):
            outcome = {
                ExitCode.COMPLETED: "completed",
                ExitCode.HANDOFF: "handoff",
                ExitCode.USER_ERROR: "failed_user",
                ExitCode.INFRA_ERROR: "failed_infra",
                ExitCode.NON_RESUMABLE: "non_resumable",
            }[ExitCode(code)]
            if outcome == "completed" and (
                status is None or status.state is not RuntimeState.COMPLETED
            ):
                return "crashed"  # exit 0 without the runtime saying so is not trusted
            return outcome
        if code is None and status is not None and status.state is not RuntimeState.RUNNING:
            return _STATUS_OUTCOME[status.state]
        return "crashed"

    def _close_segment(
        self,
        seg: Segment,
        outcome: str,
        code: int | None,
        now: float,
        message: str,
        ended: float | None = None,
    ) -> Segment:
        ran = seg.state is not SegmentState.AWAITING_USER  # a request nobody started used no GPU
        ended = now if ended is None else ended
        pointer = read_pointer(self.storage, seg.job_id)
        if pointer is not None:
            self.db.record_checkpoint(
                seg.job_id, pointer.seq, pointer.segment_id, pointer.created_at
            )
        seg = seg.model_copy(
            update={
                "state": _OUTCOME_STATE[outcome],
                "outcome": outcome,
                "ended_at": _dt(ended),
                "exit_code": code,
                "end_checkpoint_seq": pointer.seq if pointer else None,
                "message": message,
            }
        )
        self.db.put_segment(seg)
        self._beats.pop(seg.id, None)
        if ran and seg.started_at is not None:
            self.db.add_usage(seg.backend, seg.started_at.timestamp(), ended, seg.id, now)
        self._event(f"segment_{outcome}", message or outcome, now, segment=seg,
                    job=self.db.get_job(seg.job_id))  # fmt: skip
        return seg

    def _finish(
        self,
        seg: Segment,
        outcome: str,
        code: int | None,
        now: float,
        message: str,
        ended: float | None = None,
    ) -> None:
        seg = self._close_segment(seg, outcome, code, now, message, ended)
        if seg.role == "worker":
            return  # workers only help with chunks: the job follows its main segment
        job = self.db.get_job(seg.job_id)
        if job is None or job.state.is_terminal:
            return
        progressed = seg.end_checkpoint_seq != seg.start_checkpoint_seq
        if job.requested == "pause" and outcome in ("handoff", "crashed", "failed_infra"):
            self.storage.delete(control_key(job.id))
            self._set_job(job, JobState.PAUSED, now, "paused by user", requested=None)
            return
        if outcome == "completed":
            self._set_job(job, JobState.ASSEMBLING, now, "assembling outputs")
        elif outcome == "handoff":
            self._set_job(job, JobState.QUEUED, now, f"handed off on {seg.backend}")
        elif outcome in ("failed_user", "non_resumable"):
            self._set_job(job, JobState.FAILED_USER, now, message or outcome)
            self.notifier.notify("failed", f"job {job.name} failed", message or outcome)
        else:  # crashed / failed_infra
            failures = job.infra_failures + (0 if outcome == "crashed" and progressed else 1)
            if failures > self.config.scheduler.max_infra_retries:
                self._set_job(job, JobState.FAILED_INFRA, now, message or outcome,
                              infra_failures=failures)  # fmt: skip
                self.notifier.notify("failed", f"job {job.name} failed", message or outcome)
            else:
                self._set_job(job, JobState.QUEUED, now, f"{outcome} on {seg.backend}; retrying",
                              infra_failures=failures)  # fmt: skip

    # ------------------------------------------------------------------ 3. assembly

    def _assemble_completed(self, now: float) -> None:
        for job in self.db.list_jobs([JobState.ASSEMBLING]):
            try:
                result = assemble(self.storage, job.id, self.outputs_root / job.id)
            except Exception as exc:
                self._set_job(job, JobState.FAILED_INFRA, now, f"assembly failed: {exc}")
                self.notifier.notify("failed", f"job {job.name} failed", f"assembly: {exc}")
                continue
            log.info("outputs of %s: %s", job.id, result.out_dir)
            self._set_job(job, JobState.COMPLETED, now, f"outputs ready: gpurelay fetch {job.id}")
            self.notifier.notify(
                "completed", f"job {job.name} completed", f"{len(result.segments)} segment(s)"
            )

    # ------------------------------------------------------------------ 4. scheduling

    def backend_views(self, now: float) -> list[BackendView]:
        active = self.db.active_segments()
        views = []
        for slot in self.backends.values():
            quota = self.quota_state(slot, now)
            try:
                ready = slot.backend.check_ready().ready
            except Exception:
                ready = False
            views.append(
                BackendView(
                    name=slot.name,
                    automated=slot.automated,
                    ready=ready,
                    remaining_h=quota.remaining_h,
                    unlimited=slot.budget.unlimited,
                    max_concurrent=slot.max_concurrent,
                    running=sum(1 for s in active if s.backend == slot.name),
                    max_session_s=slot.max_session_s,
                    gpu_label=slot.gpu_label,
                    next_reset=quota.next_reset,
                )
            )
        return views

    def _schedule(self, now: float) -> None:
        live_jobs = {s.job_id for s in self.db.active_segments() if s.role == "main"}
        jobs = [
            j
            for j in self.db.list_jobs(SCHEDULABLE)
            if j.id not in live_jobs and j.requested is None
        ]
        if not jobs:
            return
        views = []
        for job in jobs:
            gpus = [s.gpu_model for s in self.db.segments(job.id) if s.gpu_model]
            views.append(JobView(job.id, job.priority, job.submitted_at, job.prefer_backend, gpus))
        decisions = plan(
            views, self.backend_views(now), min_segment_h=self.config.scheduler.min_segment_h
        )
        by_id = {j.id: j for j in jobs}
        for decision in decisions:
            job = by_id[decision.job_id]
            if decision.action in ("launch", "await_user"):
                assert decision.backend is not None and decision.deadline_s is not None
                self._launch(job, self.backends[decision.backend], decision.deadline_s, now)
            elif decision.action == "wait_quota":
                if job.state is not JobState.WAITING_FOR_QUOTA:
                    when = decision.wait_until.isoformat() if decision.wait_until else "unknown"
                    self._set_job(job, JobState.WAITING_FOR_QUOTA, now,
                                  f"no GPU budget left; next reset {when}",
                                  wait_until=decision.wait_until)  # fmt: skip
                    self.notifier.notify(
                        "waiting_for_quota", f"job {job.name} is waiting for GPU quota",
                        f"next reset: {when}",
                    )  # fmt: skip
            elif job.state is not JobState.QUEUED:  # wait_slot
                self._set_job(job, JobState.QUEUED, now, "waiting for a free backend slot")

    def _launch(self, job: Job, slot: BackendSlot, deadline_s: float, now: float) -> None:
        pointer = read_pointer(self.storage, job.id)
        segments = self.db.segments(job.id)
        seg = Segment(
            job_id=job.id,
            index=len(segments),
            backend=slot.name,
            state=SegmentState.LAUNCHING,
            gpu_model=slot.gpu_label,
            started_at=_dt(now),
            deadline_at=_dt(now) + timedelta(seconds=deadline_s),
            start_checkpoint_seq=pointer.seq if pointer else None,
        )
        self.db.put_segment(seg)
        self.storage.delete(control_key(job.id))
        try:
            handle = slot.backend.launch(job, seg, deadline_s)
        except BackendBusyError as exc:
            # e.g. a Kaggle session from before a controller restart is still running.
            self._close_segment(seg, "cancelled", None, now, f"not launched: {exc}")
            self.db.put_job(job.model_copy(update={"message": str(exc)}))
            return
        except Exception as exc:
            self._finish(seg, "failed_infra", None, now, f"launch failed: {exc}")
            return
        manual = not slot.automated
        seg = seg.model_copy(
            update={
                "handle_id": handle.handle_id,
                "handle_extra": dict(handle.extra),
                "state": SegmentState.AWAITING_USER if manual else SegmentState.RUNNING,
            }
        )
        self.db.put_segment(seg)
        if manual:
            self._set_job(job, JobState.AWAITING_USER, now, f"open the session on {slot.name}")
            self.notifier.notify(
                "colab_action", f"job {job.name} needs you",
                f"continue it on {slot.name}: see `gpurelay colab {job.id}`",
            )  # fmt: skip
        else:
            self._set_job(
                job, JobState.RUNNING, now, f"segment {seg.index} on {slot.name}",
                infra_failures=job.infra_failures,
            )  # fmt: skip

    # ------------------------------------------------------------------ 4b. parallel workers

    def _launch_workers(self, now: float) -> None:
        """Help parallel chunks regions with spare automated backends (Phase 9).

        Runs after normal scheduling, so queued jobs get free slots first. Workers of jobs
        that stopped running are stopped (their claimed chunks are picked up by others).
        """
        active = self.db.active_segments()
        for seg in [s for s in active if s.role == "worker"]:
            job = self.db.get_job(seg.job_id)
            if job is None or job.state.is_terminal or job.state in (
                JobState.PAUSED, JobState.ASSEMBLING
            ):  # fmt: skip
                self._stop_segment(seg, now, "cancelled")
        limit = self.config.scheduler.max_parallel_workers
        if limit <= 0:
            return
        for job in self.db.list_jobs([JobState.RUNNING]):
            try:
                self._help_region(job, limit, now)
            except Exception as exc:  # one job's problem must not stall the others
                log.warning("parallel workers for %s: %s: %s", job.id, type(exc).__name__, exc)

    def _help_region(self, job: Job, limit: int, now: float) -> None:
        segments = self.db.active_segments()
        main = next((s for s in segments if s.job_id == job.id and s.role == "main"
                     and s.state is SegmentState.RUNNING), None)  # fmt: skip
        if main is None:
            return
        status = self._read_status(job.id)
        region = status.parallel if status and status.segment_id == main.id else None
        if region is None or not region.ready:
            return
        helping = sum(1 for s in segments if s.job_id == job.id and s.region == region.region)
        pending = region.total - region.done
        for _ in range(min(limit, pending - 1) - helping):
            views = [
                v for v in self.backend_views(now)
                if v.automated and v.ready and v.free()
                and v.has_budget(self.config.scheduler.min_segment_h)
            ]  # fmt: skip
            if not views:
                return
            view = max(views, key=lambda v: (v.unlimited, v.remaining_h or 0.0, v.name))
            self._launch_worker(job, self.backends[view.name], region.region,
                                deadline_for(view), now)  # fmt: skip

    def _launch_worker(
        self, job: Job, slot: BackendSlot, region: str, deadline_s: float, now: float
    ) -> None:
        seg = Segment(
            job_id=job.id,
            index=len(self.db.segments(job.id)),
            backend=slot.name,
            state=SegmentState.LAUNCHING,
            gpu_model=slot.gpu_label,
            started_at=_dt(now),
            deadline_at=_dt(now) + timedelta(seconds=deadline_s),
            role="worker",
            region=region,
        )
        self.db.put_segment(seg)
        try:
            write_assignment(self.storage, job.id, seg.id, region)
            handle = slot.backend.launch(job, seg, deadline_s)
        except Exception as exc:
            self._close_segment(seg, "cancelled", None, now, f"worker not launched: {exc}")
            return
        seg = seg.model_copy(update={
            "handle_id": handle.handle_id, "handle_extra": dict(handle.extra),
            "state": SegmentState.RUNNING,
        })  # fmt: skip
        self.db.put_segment(seg)
        self._event("worker_started", f"worker segment {seg.index} on {slot.name}", now,
                    job=job, segment=seg)  # fmt: skip

    # ------------------------------------------------------------------ 5. budgets

    def _check_budgets(self, now: float) -> None:
        for slot in self.backends.values():
            if slot.budget.hours is None:
                continue
            quota = self.quota_state(slot, now)
            fraction = quota.fraction_left
            if fraction is not None and fraction < BUDGET_LOW_FRACTION:
                if slot.name not in self._budget_warned:
                    self._budget_warned.add(slot.name)
                    self.notifier.notify(
                        "budget_low", f"{slot.name} budget low",
                        f"{quota.remaining_h:.2f} GPU-h left of {slot.budget.hours:.1f}",
                    )  # fmt: skip
            else:
                self._budget_warned.discard(slot.name)


_ACTIVE = (SegmentState.LAUNCHING, SegmentState.RUNNING, SegmentState.AWAITING_USER)
