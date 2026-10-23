"""Controller flows with fake backends and a fake clock (plan Phase 5 verification)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from fakes import FakeBackend
from gpurelay.config import Config, NotifySettings, SchedulerSettings
from gpurelay.controller import daemon as daemon_mod
from gpurelay.controller.daemon import Controller
from gpurelay.controller.db import StateDB
from gpurelay.controller.notify import Notifier
from gpurelay.controller.quota import WEEK_S, Budget
from gpurelay.controller.registry import BackendSlot
from gpurelay.controller.views import render_status
from gpurelay.models import (
    BudgetModel,
    CheckpointPointer,
    Job,
    JobKind,
    JobState,
    Progress,
    Rate,
    RuntimeState,
    SegmentState,
    StatusFile,
)
from gpurelay.storage import LocalFSStorage
from gpurelay.storage.layout import control_key, latest_key, status_key

H = 3600.0
T0 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> float:
        return self.now


@dataclass
class Harness:
    controller: Controller
    db: StateDB
    storage: LocalFSStorage
    clock: Clock
    backends: dict[str, FakeBackend]
    notifier: Notifier

    def submit(self, name: str = "job", **fields: Any) -> Job:
        job = Job(name=name, kind=JobKind.SCRIPT, entry=f"{name}.py", **fields)
        return self.db.put_job(job)

    def job(self, job_id: str) -> Job:
        job = self.db.get_job(job_id)
        assert job is not None
        return job

    def live(self, job_id: str) -> list[Any]:
        return [s for s in self.db.active_segments() if s.job_id == job_id]

    def finish(self, job_id: str, code: int, state: str, message: str = "") -> None:
        [seg] = self.live(job_id)
        self.backends[seg.backend].finish(job_id, seg.id, code, state, message)

    def commit(self, job_id: str, seq: int, segment_id: str) -> None:
        pointer = CheckpointPointer(
            seq=seq, files=[], sha256s={}, safe_point_id="x",
            created_at=datetime.fromtimestamp(self.clock.now, timezone.utc), segment_id=segment_id,
        )  # fmt: skip
        self.storage.put_bytes(pointer.model_dump_json().encode(), latest_key(job_id))


def _harness(tmp_path: Path, specs: dict[str, tuple[float | None, bool, int, str]]) -> Harness:
    """specs: name -> (weekly budget hours or None=unlimited, automated, max_concurrent, gpu)."""
    storage = LocalFSStorage(tmp_path / "store")
    db = StateDB(tmp_path / "state.db")
    clock = Clock()
    fakes: dict[str, FakeBackend] = {}
    slots: dict[str, BackendSlot] = {}
    for name, (hours, automated, concurrent, gpu) in specs.items():
        fake = FakeBackend(name, storage, automated=automated)
        fakes[name] = fake
        model = BudgetModel.WEEKLY_ROLLING if automated else BudgetModel.UNKNOWN_OPPORTUNISTIC
        slots[name] = BackendSlot(
            name=name,
            backend=fake,
            budget=Budget(model, hours, unlimited=automated and hours is None, estimate=False),
            automated=automated,
            max_concurrent=concurrent,
            max_session_s=12 * H,
            gpu_label=gpu,
        )
    config = Config(
        scheduler=SchedulerSettings(min_segment_h=0.5, max_infra_retries=2),
        notify=NotifySettings(desktop=False),
    )
    notifier = Notifier(config.notify)
    controller = Controller(
        config=config, db=db, storage=storage, backends=slots,
        outputs_root=tmp_path / "outputs", notifier=notifier, clock=clock,
    )  # fmt: skip
    return Harness(controller, db, storage, clock, fakes, notifier)


@pytest.fixture(autouse=True)
def stub_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class Result:
        out_dir: Path
        segments: list[str]

    monkeypatch.setattr(daemon_mod, "assemble", lambda storage, job_id, out: Result(out, ["s"]))


# ---------------------------------------------------------------- budget lifecycle


def test_budget_exhaustion_waits_for_reset_then_resumes(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (1.0, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    assert h.job(job.id).state is JobState.RUNNING
    assert h.backends["kaggle"].launched[0][2] == pytest.approx(1 * H)  # deadline <= budget

    h.clock.now += 1 * H
    h.finish(job.id, 85, "handoff")
    h.controller.tick()
    waiting = h.job(job.id)
    assert waiting.state is JobState.WAITING_FOR_QUOTA
    assert waiting.wait_until == datetime.fromtimestamp(T0 + H + WEEK_S, timezone.utc)
    assert [n.kind for n in h.notifier.sent].count("waiting_for_quota") == 1

    h.clock.now = T0 + 3 * 24 * H
    h.controller.tick()
    assert h.job(job.id).state is JobState.WAITING_FOR_QUOTA  # still inside the window

    h.clock.now = T0 + H + WEEK_S + 1
    h.controller.tick()
    assert h.job(job.id).state is JobState.RUNNING
    assert len(h.backends["kaggle"].launched) == 2


def test_ledger_matches_segment_durations(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (30.0, True, 1, "T4")})
    job = h.submit()
    for minutes in (30, 45, 20):
        h.controller.tick()
        h.clock.now += minutes * 60
        h.finish(job.id, 85, "handoff")
        h.controller.tick()
    usage = [e.hours for e in h.db.ledger("a") if e.kind == "usage"]
    durations = [
        (s.ended_at.timestamp() - s.started_at.timestamp()) / H  # type: ignore[union-attr]
        for s in h.db.segments(job.id)
        if s.ended_at
    ]
    assert usage == pytest.approx(durations)
    assert durations == pytest.approx([0.5, 0.75, 1 / 3])


def test_budget_low_notification(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (10.0, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    h.clock.now += 9.5 * H
    h.controller.tick()  # live usage counts
    assert "budget_low" in [n.kind for n in h.notifier.sent]
    del job


# ---------------------------------------------------------------- placement


def test_parallel_jobs_on_different_backends_and_one_segment_per_job(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (10.0, True, 1, "T4"), "b": (8.0, True, 1, "P100")})
    first = h.submit("first")
    h.clock.now += 1
    second = h.submit("second")
    h.clock.now += 1
    third = h.submit("third")
    h.controller.tick()
    assert [s.backend for s in h.live(first.id)] == ["a"]  # most budget
    assert [s.backend for s in h.live(second.id)] == ["b"]
    assert h.live(third.id) == [] and h.job(third.id).state is JobState.QUEUED
    for _ in range(3):
        h.controller.tick()
    assert all(len(h.live(j.id)) <= 1 for j in (first, second, third))
    assert len(h.backends["a"].launched) == len(h.backends["b"].launched) == 1


def test_same_gpu_preferred_when_budgets_tie(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (10.0, True, 1, "T4"), "b": (10.0, True, 1, "P100")})
    job = h.submit()
    h.controller.tick()
    first_backend = h.live(job.id)[0].backend
    h.clock.now += 60
    h.finish(job.id, 85, "handoff")
    h.controller.tick()  # budgets now differ by 1 minute: still a tie
    assert h.live(job.id)[0].backend == first_backend


def test_manual_backend_means_awaiting_user(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (0.0, True, 1, "T4"), "colab": (None, False, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    assert h.job(job.id).state is JobState.AWAITING_USER
    assert "colab_action" in [n.kind for n in h.notifier.sent]
    [seg] = h.live(job.id)
    assert seg.state is SegmentState.AWAITING_USER
    status = StatusFile(job_id=job.id, segment_id=seg.id, state=RuntimeState.RUNNING,
                        heartbeat_at=datetime.fromtimestamp(h.clock.now, timezone.utc))  # fmt: skip
    h.storage.put_bytes(status.model_dump_json().encode(), status_key(job.id))
    h.controller.tick()
    assert h.job(job.id).state is JobState.RUNNING


def _user_starts(h: Harness, job_id: str, state: RuntimeState = RuntimeState.RUNNING) -> None:
    [seg] = h.live(job_id)
    status = StatusFile(job_id=job_id, segment_id=seg.id, state=state,
                        heartbeat_at=datetime.fromtimestamp(h.clock.now, timezone.utc))  # fmt: skip
    h.storage.put_bytes(status.model_dump_json().encode(), status_key(job_id))


def test_manual_session_time_counts_from_when_the_user_starts(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"colab": (None, False, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    [requested] = h.live(job.id)
    h.clock.now += 3 * H  # the user opens the notebook three hours later
    _user_starts(h, job.id)
    h.controller.tick()
    [seg] = h.live(job.id)
    assert seg.state is SegmentState.RUNNING
    assert seg.started_at is not None and seg.started_at.timestamp() == h.clock.now
    assert requested.deadline_at is not None and seg.deadline_at is not None
    assert seg.deadline_at - seg.started_at == requested.deadline_at - requested.started_at
    h.clock.now += 1 * H
    _user_starts(h, job.id, RuntimeState.HANDOFF)  # the session hands off
    h.controller.tick()
    usage = [e for e in h.db.ledger("colab") if e.kind == "usage"]
    assert sum(e.hours or 0 for e in usage) == pytest.approx(1.0)
    # The job needs Colab again: a new request and a new notification.
    h.controller.tick()
    [again] = h.live(job.id)
    assert again.id != seg.id and again.state is SegmentState.AWAITING_USER
    assert [n.kind for n in h.notifier.sent].count("colab_action") == 2


def test_cancelling_an_unstarted_request_uses_no_gpu_time(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"colab": (None, False, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    [seg] = h.live(job.id)
    h.clock.now += 5 * H
    h.db.put_job(h.job(job.id).model_copy(update={"requested": "cancel"}))
    h.controller.tick()
    assert h.job(job.id).state is JobState.CANCELLED
    assert h.backends["colab"].cancelled == [seg.id]
    assert not [e for e in h.db.ledger("colab") if e.kind == "usage"]


# ---------------------------------------------------------------- outcomes


def test_completion_assembles_and_notifies(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    h.finish(job.id, 0, "completed")
    h.controller.tick()
    assert h.job(job.id).state is JobState.COMPLETED
    assert "completed" in [n.kind for n in h.notifier.sent]
    assert not h.controller.busy()


def test_user_failure_is_not_retried(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    h.finish(job.id, 86, "failed_user", "ValueError: boom")
    h.controller.tick()
    h.controller.tick()
    assert h.job(job.id).state is JobState.FAILED_USER
    assert h.job(job.id).message == "ValueError: boom"
    assert len(h.backends["a"].launched) == 1


def test_infra_failures_retry_then_give_up(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    for _ in range(3):
        h.controller.tick()
        h.finish(job.id, 87, "failed_infra")
        h.controller.tick()
    assert h.job(job.id).state is JobState.FAILED_INFRA  # 3 failures > max_infra_retries=2
    assert len(h.backends["a"].launched) == 3


def test_crashes_with_progress_do_not_count_as_failures(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    for seq in range(1, 6):
        h.controller.tick()
        [seg] = h.live(job.id)
        h.commit(job.id, seq, seg.id)
        h.backends["a"].finish(job.id, seg.id, -9, "running")  # killed: status says running
        h.controller.tick()
    assert h.job(job.id).infra_failures == 0
    assert h.job(job.id).state is JobState.RUNNING  # relaunched every time
    assert [s.outcome for s in h.db.segments(job.id) if s.outcome] == ["crashed"] * 5


# ---------------------------------------------------------------- user requests


def test_pause_resume_cancel(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    h.db.put_job(h.job(job.id).model_copy(update={"requested": "pause"}))
    h.controller.tick()
    control = h.storage.get_bytes(control_key(job.id))
    assert control is not None and b'"handoff": true' in control
    h.finish(job.id, 85, "handoff")
    h.controller.tick()
    assert h.job(job.id).state is JobState.PAUSED
    assert h.storage.get_bytes(control_key(job.id)) is None
    h.controller.tick()
    assert len(h.backends["a"].launched) == 1  # paused jobs are not scheduled

    h.db.put_job(h.job(job.id).model_copy(update={"requested": "resume"}))
    h.controller.tick()
    assert h.job(job.id).state is JobState.RUNNING

    h.db.put_job(h.job(job.id).model_copy(update={"requested": "cancel"}))
    h.controller.tick()
    assert h.job(job.id).state is JobState.CANCELLED
    assert h.backends["a"].cancelled == [h.db.segments(job.id)[-1].id]


# ---------------------------------------------------------------- status view


def test_status_snapshot(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4"), "colab": (None, False, 1, "T4")})
    job = h.submit("vae", id="job-000000000001")
    h.controller.tick()
    [seg] = h.live(job.id)
    h.clock.now += 2 * H
    status = StatusFile(
        job_id=job.id, segment_id=seg.id, state=RuntimeState.RUNNING,
        heartbeat_at=datetime.fromtimestamp(h.clock.now, timezone.utc),
        progress=Progress(loop={"L61-abcdef": (400, 1000)}), rate=Rate(unit="iter", per_hour=200),
    )  # fmt: skip
    h.storage.put_bytes(status.model_dump_json().encode(), status_key(job.id))
    console = Console(
        record=True, width=160, color_system=None, legacy_windows=False, safe_box=False
    )
    render_status(console, h.controller, h.clock.now)
    text = console.export_text()
    assert "400/1000" in text and "~3.00h (progress)" in text  # ETA = 600 iters / 200 per h
    assert "28.00h" in text  # 30h budget - 2h live usage
    assert "unknown" in text  # colab: manual, unknown budget
    snapshot = Path(__file__).parent / "snapshots" / "status.txt"
    if os.environ.get("GPURELAY_UPDATE_SNAPSHOTS") or not snapshot.exists():
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_text(text, encoding="utf-8")
    assert text == snapshot.read_text(encoding="utf-8")


# ---------------------------------------------------------------- hardening


def _beat(h: Harness, job_id: str, at: float) -> None:
    [seg] = h.live(job_id)
    status = StatusFile(job_id=job_id, segment_id=seg.id, state=RuntimeState.RUNNING,
                        heartbeat_at=datetime.fromtimestamp(at, timezone.utc))  # fmt: skip
    h.storage.put_bytes(status.model_dump_json().encode(), status_key(job_id))


@pytest.mark.parametrize("skew_s", [-3600.0, 0.0, 3600.0], ids=["behind", "in-sync", "ahead"])
def test_heartbeat_staleness_ignores_clock_skew(tmp_path: Path, skew_s: float) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    limit = h.controller.config.runtime.stale_heartbeat_s * 2  # running sessions: factor 2
    job = h.submit()
    h.controller.tick()
    for _ in range(4):  # alive: the (skewed) heartbeat keeps changing
        _beat(h, job.id, h.clock.now + skew_s)
        h.controller.tick()
        h.clock.now += limit * 0.6
    assert h.job(job.id).state is JobState.RUNNING and h.backends["a"].cancelled == []
    h.clock.now += limit + 1  # the heartbeat stopped changing
    h.controller.tick()
    assert h.backends["a"].cancelled != []
    assert h.db.segments(job.id)[0].outcome == "crashed"


def test_controller_survives_outages_with_backoff_and_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    h.submit()
    real_schedule = h.controller._schedule
    calls = {"n": 0}

    def flaky(now: float) -> None:
        calls["n"] += 1
        if calls["n"] <= 6:
            raise ConnectionError("hub unreachable")
        real_schedule(now)

    monkeypatch.setattr(h.controller, "_schedule", flaky)
    sleeps: list[float] = []
    h.controller.run(interval_s=1.0, max_ticks=8, sleep=sleeps.append)
    assert sleeps == [1, 2, 4, 8, 8, 8, 1]
    assert [n.kind for n in h.notifier.sent].count("controller_error") == 1
    assert len(h.backends["a"].launched) == 1  # recovered and scheduled the job


def test_one_broken_segment_does_not_block_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4"), "b": (None, True, 1, "T4")})
    bad, good = h.submit("bad"), h.submit("good")
    h.controller.tick()
    real_read = h.controller._read_status

    def read(job_id: str, segment: Any = None) -> StatusFile | None:
        if job_id == bad.id:
            raise OSError("corrupt status")
        return real_read(job_id, segment)

    monkeypatch.setattr(h.controller, "_read_status", read)
    h.finish(good.id, 0, "completed")
    h.controller.tick()
    assert h.job(good.id).state in (JobState.ASSEMBLING, JobState.COMPLETED)
    assert h.job(bad.id).state is JobState.RUNNING
