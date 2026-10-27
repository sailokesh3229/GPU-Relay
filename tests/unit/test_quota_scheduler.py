"""Quota ledger windows/corrections and the pure scheduling policy (spec §13.2, §13.4)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpurelay.controller.db import StateDB
from gpurelay.controller.estimator import BudgetLine, fit_message
from gpurelay.controller.quota import WEEK_S, Budget, LiveUsage, QuotaLedger
from gpurelay.controller.scheduler import BackendView, JobView, plan
from gpurelay.models import BudgetModel, Job, JobKind, JobState, Segment

H = 3600.0
T0 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def db(tmp_path: Path) -> StateDB:
    return StateDB(tmp_path / "state.db")


# ---------------------------------------------------------------- database


def test_db_round_trip_and_migration(db: StateDB, tmp_path: Path) -> None:
    assert db.schema_version == 1
    job = db.put_job(Job(name="a", kind=JobKind.SCRIPT, entry="a.py"))
    assert db.get_job(job.id) == job
    assert db.find_job(job.id[:8]) == job
    seg = Segment(job_id=job.id, index=0, backend="b")
    db.put_segment(seg)
    assert db.segments(job.id) == [seg]
    assert db.active_segments() == [seg]
    db.add_event("x", "hello", now=T0, job_id=job.id)
    assert db.events(job.id)[0].message == "hello"
    db.close()
    assert StateDB(tmp_path / "state.db").schema_version == 1  # reopen: no re-migration


def test_list_jobs_by_state(db: StateDB) -> None:
    a = db.put_job(Job(name="a", kind=JobKind.SCRIPT, entry="a.py"))
    db.put_job(Job(name="b", kind=JobKind.SCRIPT, entry="b.py", state=JobState.COMPLETED))
    assert [j.id for j in db.list_jobs([JobState.QUEUED])] == [a.id]


# ---------------------------------------------------------------- quota


def test_weekly_rolling_window(db: StateDB) -> None:
    budget = Budget(BudgetModel.WEEKLY_ROLLING, 30.0)
    ledger = QuotaLedger(db)
    db.add_usage("k", T0 - 8 * 24 * H, T0 - 8 * 24 * H + 5 * H, "old", T0)  # outside window
    db.add_usage("k", T0 - 2 * H, T0, "s1", T0)
    state = ledger.state("k", budget, T0, [LiveUsage(T0 - H, T0)])
    assert state.used_h == pytest.approx(3.0)
    assert state.remaining_h == pytest.approx(27.0)
    assert state.next_reset is None  # enough budget left


def test_rolling_reset_time_when_exhausted(db: StateDB) -> None:
    budget = Budget(BudgetModel.WEEKLY_ROLLING, 4.0)
    db.add_usage("k", T0 - 3 * 24 * H, T0 - 3 * 24 * H + 1 * H, "a", T0)
    db.add_usage("k", T0 - 4 * H, T0, "b", T0)
    state = QuotaLedger(db).state("k", budget, T0, min_segment_h=0.5)
    assert state.remaining_h == 0
    # The oldest hour leaves the window 7 days after it ended.
    assert state.next_reset == datetime.fromtimestamp(T0 - 3 * 24 * H + H + WEEK_S, timezone.utc)


def test_monthly_window_resets_on_the_first(db: StateDB) -> None:
    budget = Budget(BudgetModel.MONTHLY, 80.0)
    db.add_usage("l", T0 - 15 * 24 * H, T0 - 15 * 24 * H + 10 * H, "feb", T0)  # previous month
    db.add_usage("l", T0 - 5 * H, T0, "mar", T0)
    state = QuotaLedger(db).state("l", budget, T0)
    assert state.remaining_h == pytest.approx(75.0)
    assert state.next_reset == datetime(2026, 4, 1, tzinfo=timezone.utc)


def test_user_corrections(db: StateDB) -> None:
    budget = Budget(BudgetModel.WEEKLY_ROLLING, 30.0)
    ledger = QuotaLedger(db)
    db.add_correction("k", remaining_h=12.0, used_h=None, now=T0 - 2 * H)
    db.add_usage("k", T0 - H, T0, "s", T0)
    state = ledger.state("k", budget, T0)
    assert state.remaining_h == pytest.approx(11.0) and state.corrected
    db.add_correction("k", remaining_h=None, used_h=20.0, now=T0)
    assert ledger.state("k", budget, T0).remaining_h == pytest.approx(10.0)


def test_unlimited_and_unknown_budgets(db: StateDB) -> None:
    ledger = QuotaLedger(db)
    local = ledger.state("local", Budget(BudgetModel.UNKNOWN_OPPORTUNISTIC, None, True), T0)
    assert local.remaining_h is None and local.next_reset is None
    colab = ledger.state("colab", Budget(BudgetModel.UNKNOWN_OPPORTUNISTIC, None), T0)
    assert colab.remaining_h is None


# ---------------------------------------------------------------- scheduler


def _backend(name: str, remaining: float | None, **kw: object) -> BackendView:
    defaults: dict[str, object] = dict(
        automated=True, ready=True, unlimited=False, max_concurrent=1, running=0,
        max_session_s=12 * H, gpu_label=None, next_reset=None,
    )  # fmt: skip
    defaults.update(kw)
    return BackendView(name=name, remaining_h=remaining, **defaults)  # type: ignore[arg-type]


def _job(job_id: str, priority: int = 0, minute: int = 0, **kw: object) -> JobView:
    submitted = datetime(2026, 3, 1, 0, minute, tzinfo=timezone.utc)
    return JobView(job_id, priority, submitted, **kw)  # type: ignore[arg-type]


def test_priority_then_submission_order_and_most_budget() -> None:
    backends = [_backend("kaggle", 20.0), _backend("sim", 5.0)]
    decisions = plan([_job("late", minute=5), _job("early"), _job("urgent", priority=1)],
                     backends, min_segment_h=0.5)  # fmt: skip
    assert [(d.job_id, d.backend) for d in decisions] == [
        ("urgent", "kaggle"),
        ("early", "sim"),
        ("late", None),
    ]
    assert decisions[2].action == "wait_slot"


def test_deadline_never_exceeds_remaining_budget() -> None:
    [decision] = plan([_job("j")], [_backend("k", 2.0)], min_segment_h=0.5)
    assert decision.deadline_s == pytest.approx(2 * H)


def test_preferred_backend_and_gpu_tie_break() -> None:
    backends = [_backend("a", 10.0, gpu_label="T4"), _backend("b", 10.2, gpu_label="P100")]
    assert plan([_job("j", previous_gpus=["T4"])], backends, min_segment_h=0.5)[0].backend == "a"
    assert plan([_job("j")], backends, min_segment_h=0.5)[0].backend == "b"
    assert plan([_job("j", prefer_backend="a")], backends, min_segment_h=0.5)[0].backend == "a"


def test_concurrency_limits_and_parallel_jobs() -> None:
    backends = [_backend("a", 10.0, max_concurrent=2), _backend("b", 1.0, running=1)]
    decisions = plan([_job("1"), _job("2", minute=1), _job("3", minute=2)], backends,
                     min_segment_h=0.5)  # fmt: skip
    assert [d.backend for d in decisions] == ["a", "a", None]


def test_manual_only_means_awaiting_user() -> None:
    backends = [_backend("kaggle", 0.0), _backend("colab", None, automated=False)]
    [decision] = plan([_job("j")], backends, min_segment_h=0.5)
    assert (decision.action, decision.backend) == ("await_user", "colab")


def test_manual_backend_set_first_comes_before_automated_ones() -> None:
    kaggle, colab = _backend("kaggle", 20.0), _backend("colab", None, automated=False)
    [decision] = plan([_job("j")], [kaggle, colab], min_segment_h=0.5)
    assert decision.backend == "kaggle"  # default: Colab only when nothing automated can
    colab.first = True
    first, second = plan([_job("j"), _job("k", minute=1)], [kaggle, colab], min_segment_h=0.5)
    assert (first.action, first.backend) == ("await_user", "colab")
    assert (second.action, second.backend) == ("launch", "kaggle")  # Colab has one slot
    colab.ready = False  # on hold after nobody pressed Run all
    [decision] = plan([_job("j")], [kaggle, colab], min_segment_h=0.5)
    assert decision.backend == "kaggle"


def test_no_budget_waits_for_earliest_reset() -> None:
    reset_a = datetime(2026, 3, 5, tzinfo=timezone.utc)
    reset_b = datetime(2026, 3, 3, tzinfo=timezone.utc)
    backends = [_backend("a", 0.1, next_reset=reset_a), _backend("b", 0.0, next_reset=reset_b)]
    [decision] = plan([_job("j")], backends, min_segment_h=0.5)
    assert decision.action == "wait_quota" and decision.wait_until == reset_b


def test_unready_backends_are_skipped() -> None:
    [decision] = plan([_job("j")], [_backend("a", 10.0, ready=False)], min_segment_h=0.5)
    assert decision.action == "wait_quota"


# ---------------------------------------------------------------- estimator message


def test_fit_message() -> None:
    lines = [BudgetLine("kaggle", 22.0, 30.0, True), BudgetLine("colab", None, None, False)]
    assert fit_message(10, lines).endswith("Fits this week: kaggle 10.0h")
    assert fit_message(30, lines).endswith("Fits this week: kaggle 22.0h + colab (manual) ~8.0h")
    assert "spans ~4 week(s)" in fit_message(120, [BudgetLine("kaggle", 0.0, 30.0, True)])
    assert "no quota" in fit_message(5, [BudgetLine("local", None, None, True, True)])
    assert "unknown" in fit_message(None, lines)
