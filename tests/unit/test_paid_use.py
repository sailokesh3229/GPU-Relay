"""Paid GPU time (plan P16): free first, paid only within the job's allowance and the
platform's monthly limit. Both are $0 by default, so nothing is ever paid until the owner
raises them."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_controller import _harness

from gpurelay.controller.scheduler import BackendView, JobView, plan
from gpurelay.controller.spend import backend_spent_this_month, job_spent, segment_cost
from gpurelay.models import JobState, Segment

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)


def view(name: str, free_h: float, *, automated: bool = True, paid: float | None = None,
         paid_left: float = 0.0, speed: float = 1.0) -> BackendView:  # fmt: skip
    return BackendView(
        name=name, automated=automated, ready=True, remaining_h=free_h, unlimited=False,
        max_concurrent=1, running=0, max_session_s=4 * 3600, budget_model="weekly_rolling",
        speed=speed, paid_usd_per_h=paid, paid_left_usd=paid_left,
    )  # fmt: skip


def job(paid_left: float = 0.0) -> JobView:
    return JobView("j", 0, NOW, paid_left_usd=paid_left)


def test_by_default_nothing_is_ever_paid() -> None:
    empty_free = view("kaggle", 0.0)
    modal = view("modal", 0.0, paid=0.59, paid_left=0.0)  # monthly limit $0
    [d] = plan([job(paid_left=50.0)], [empty_free, modal], min_segment_h=0.5)
    assert d.action == "wait_quota" and not d.paid
    modal = view("modal", 0.0, paid=0.59, paid_left=20.0)
    [d] = plan([job(paid_left=0.0)], [empty_free, modal], min_segment_h=0.5)  # job: free only
    assert d.action == "wait_quota"


def test_free_time_always_comes_first() -> None:
    kaggle = view("kaggle", 10.0)
    modal = view("modal", 0.0, paid=0.10, paid_left=20.0, speed=5.0)  # cheap and fast, but paid
    [d] = plan([job(paid_left=20.0)], [kaggle, modal], min_segment_h=0.5)
    assert (d.action, d.backend, d.paid) == ("launch", "kaggle", False)


def test_colab_free_but_manual_comes_before_paid() -> None:
    colab = view("colab_manual", 10.0, automated=False)
    modal = view("modal", 0.0, paid=0.59, paid_left=20.0)
    [d] = plan([job(paid_left=20.0)], [view("kaggle", 0.0), colab, modal], min_segment_h=0.5)
    assert (d.action, d.backend) == ("await_user", "colab_manual")


def test_paid_time_is_the_cheapest_per_unit_of_work_and_never_passes_a_limit() -> None:
    t4 = view("cheap-t4", 0.0, paid=0.59, paid_left=100.0, speed=1.0)
    a100 = view("fast-a100", 0.0, paid=2.10, paid_left=100.0, speed=5.0)  # 0.42 per T4-hour
    [d] = plan([job(paid_left=1.05)], [t4, a100], min_segment_h=0.1)
    assert (d.action, d.backend, d.paid) == ("launch", "fast-a100", True)
    assert d.deadline_s == pytest.approx(1.05 / 2.10 * 3600)  # half an hour: the job's $1.05
    tight = view("cheap-t4", 0.0, paid=0.59, paid_left=0.30)  # platform limit nearly used
    [d] = plan([job(paid_left=10.0)], [tight], min_segment_h=0.25)
    assert d.paid and d.deadline_s == pytest.approx(0.30 / 0.59 * 3600)
    [d] = plan([job(paid_left=10.0)], [tight], min_segment_h=1.0)  # $0.30 < a 1 h segment
    assert d.action == "wait_quota"


def test_costs_are_price_times_time_and_monthly() -> None:
    start = NOW - timedelta(hours=2)
    seg = Segment(job_id="j", index=0, backend="modal", started_at=start,
                  ended_at=start + timedelta(minutes=30), paid_usd_per_h=0.60)  # fmt: skip
    assert segment_cost(seg, NOW.timestamp()) == pytest.approx(0.30)
    free = seg.model_copy(update={"paid_usd_per_h": None})
    assert segment_cost(free, NOW.timestamp()) == 0.0
    running = seg.model_copy(update={"ended_at": None})
    assert segment_cost(running, NOW.timestamp()) == pytest.approx(1.20)  # so far


def test_the_controller_launches_paid_time_only_when_allowed(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"modal": (0.0, True, 1, "T4")})  # no free time left
    slot = h.controller.backends["modal"]
    slot.paid_usd_per_h, slot.paid_cap_usd = 0.59, 5.0
    free_only = h.submit("free-only")
    h.controller.tick()
    assert h.job(free_only.id).state is JobState.WAITING_FOR_QUOTA  # allowance $0
    allowed = h.submit("allowed", max_paid_usd=2.0)
    h.controller.tick()
    j = h.job(allowed.id)
    assert j.state is JobState.RUNNING and "paid" in j.message
    [seg] = h.db.segments(allowed.id)
    assert seg.paid_usd_per_h == 0.59
    assert any(n.kind == "paid_gpu" for n in h.notifier.sent)
    h.clock.now += 3600
    now = h.clock()
    assert job_spent(h.db, allowed.id, now) == pytest.approx(0.59)
    assert backend_spent_this_month(h.db, "modal", now) == pytest.approx(0.59)
