"""Credit left on the user's paid platforms (spec §21.2-21.3, plan G2): free first, then the
credit they bought (cheapest per unit of work), then card only within their limits; never
past what is left; the platform's own balance when it reports one; told when it gets low."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_controller import _harness

from gpurelay.controller.credits import card_spent_this_month, credit_state, spent_since
from gpurelay.controller.scheduler import BackendView, JobView, plan
from gpurelay.models import JobState, Segment

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def view(name: str, free_h: float = 0.0, *, price: float | None = None, credit: float = 0.0,
         card_left: float = 0.0, speed: float = 1.0) -> BackendView:  # fmt: skip
    return BackendView(
        name=name, automated=True, ready=True, remaining_h=free_h, unlimited=False,
        max_concurrent=1, running=0, max_session_s=6 * 3600, budget_model="monthly",
        speed=speed, paid_usd_per_h=price, paid_left_usd=card_left, credit_left_usd=credit,
    )  # fmt: skip


def job(paid_left: float = 0.0) -> JobView:
    return JobView("j", 0, NOW, paid_left_usd=paid_left)


def test_free_first_then_credit_then_card() -> None:
    kaggle = view("kaggle", free_h=10.0)
    runpod = view("runpod", price=0.69, credit=20.0)
    [d] = plan([job()], [kaggle, runpod], min_segment_h=0.5)
    assert (d.backend, d.paid) == ("kaggle", False)
    [d] = plan([job()], [view("kaggle"), runpod], min_segment_h=0.5)
    assert (d.backend, d.paid, d.paid_from) == ("runpod", True, "credit")  # no job allowance
    card = view("aws", price=0.53, card_left=50.0)  # cheaper, but by card
    [d] = plan([job(paid_left=10.0)], [view("kaggle"), runpod, card], min_segment_h=0.5)
    assert (d.backend, d.paid_from) == ("runpod", "credit")  # credit is already paid for
    [d] = plan([job(paid_left=10.0)], [view("kaggle"), card], min_segment_h=0.5)
    assert (d.backend, d.paid_from) == ("aws", "card")
    [d] = plan([job()], [view("kaggle"), card], min_segment_h=0.5)  # card needs an allowance
    assert d.action == "wait_quota"


def test_credit_goes_to_the_cheapest_per_unit_of_work_and_never_past_what_is_left() -> None:
    t4 = view("gcp", price=0.54, credit=100.0, speed=1.0)
    h100 = view("runpod", price=2.69, credit=100.0, speed=9.0)  # 0.30 per T4-hour
    [d] = plan([job()], [t4, h100], min_segment_h=0.5)
    assert d.backend == "runpod"
    low = view("vastai", price=0.40, credit=1.00)
    [d] = plan([job()], [low], min_segment_h=0.5)
    assert d.paid_from == "credit" and d.deadline_s == pytest.approx(1.00 / 0.40 * 3600)
    [d] = plan([job()], [view("vastai", price=0.40, credit=0.10)], min_segment_h=0.5)
    assert d.action == "wait_quota"  # $0.10 is less than half an hour


def _seg(backend: str, start: datetime, hours: float | None, price: float,
         paid_from: str = "credit") -> Segment:  # fmt: skip
    return Segment(job_id="j", index=0, backend=backend, started_at=start,
                   ended_at=None if hours is None else start + timedelta(hours=hours),
                   paid_usd_per_h=price, paid_from=paid_from)  # type: ignore[arg-type]  # fmt: skip


def test_credit_counts_down_from_the_amount_or_the_platforms_balance(tmp_path: Path) -> None:
    h = _harness(tmp_path, {})
    now = NOW.timestamp()
    entered_at = (NOW - timedelta(hours=10)).timestamp()
    h.db.put_segment(_seg("runpod", NOW - timedelta(hours=12), 1.0, 1.0))  # before: not counted
    h.db.put_segment(_seg("runpod", NOW - timedelta(hours=5), 2.0, 1.5))  # $3
    h.db.put_segment(_seg("runpod", NOW - timedelta(hours=1), None, 2.0))  # running: $2 so far
    state = credit_state(h.db, "runpod", now, entered_usd=20.0, entered_at=entered_at)
    assert state and state.left_usd == pytest.approx(15.0) and state.source == "estimated"
    # The platform said $12 an hour ago: only what ran since counts ($2 of the running one).
    reported = credit_state(h.db, "runpod", now, entered_usd=20.0, entered_at=entered_at,
                            balance=(12.0, now - 3600))  # fmt: skip
    assert reported and reported.left_usd == pytest.approx(10.0) and reported.source == "platform"
    assert credit_state(h.db, "lambda", now, entered_usd=None, entered_at=None) is None
    assert spent_since(h.db, "runpod", now - 1800, now) == pytest.approx(1.0)


def test_card_limits_do_not_count_credit_sessions(tmp_path: Path) -> None:
    h = _harness(tmp_path, {})
    h.db.put_segment(_seg("aws", NOW - timedelta(hours=3), 1.0, 0.50, "credit"))
    h.db.put_segment(_seg("aws", NOW - timedelta(hours=2), 1.0, 0.50, "card"))
    assert card_spent_this_month(h.db, "aws", NOW.timestamp()) == pytest.approx(0.50)


def test_the_controller_spends_credit_after_free_time_and_says_when_it_runs_low(
    tmp_path: Path,
) -> None:
    h = _harness(tmp_path, {"kaggle": (0.0, True, 1, "T4"), "runpod": (0.0, True, 1, "RTX 4090")})
    slot = h.controller.backends["runpod"]
    slot.paid_usd_per_h, slot.paid_only = 0.70, True
    slot.credit_usd, slot.credit_set_at = 3.50, h.clock.now
    first = h.submit("train")
    h.controller.tick()
    j = h.job(first.id)
    assert j.state is JobState.RUNNING and "your credit" in j.message
    [seg] = h.db.segments(first.id)
    assert (seg.paid_from, seg.paid_usd_per_h) == ("credit", 0.70)
    _, _, deadline = h.backends["runpod"].launched[-1]
    assert deadline == pytest.approx(min(12 * 3600, 3.50 / 0.70 * 3600))  # 5 h of credit
    h.clock.now += 4.6 * 3600  # $3.22 used: $0.28 left, under one half-hour session
    h.controller.tick()
    sent = [n for n in h.notifier.sent if n.kind == "credit_low"]
    assert len(sent) == 1 and "used up" in sent[0].title
    h.controller.tick()
    assert len([n for n in h.notifier.sent if n.kind == "credit_low"]) == 1  # said once
    # (no heartbeat in all that time: the session counted as lost) the job is not started on
    # RunPod again: what is left does not pay for a session
    assert not h.live(first.id) and len(h.backends["runpod"].launched) == 1
    assert h.job(first.id).state is JobState.WAITING_FOR_QUOTA


def test_credit_is_used_only_while_the_user_allows_it(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"runpod": (0.0, True, 1, "RTX 4090")})
    slot = h.controller.backends["runpod"]
    slot.paid_usd_per_h, slot.paid_only, slot.credit_usd = 0.70, True, 50.0
    slot.use_credits = False
    queued = h.submit("train")
    h.controller.tick()
    assert h.job(queued.id).state is JobState.WAITING_FOR_QUOTA and not h.live(queued.id)


def test_a_platform_balance_is_read_at_most_every_ten_minutes(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"vastai": (0.0, True, 1, "RTX 4090")})
    calls: list[float] = []

    def balance() -> float:
        calls.append(h.clock.now)
        return 7.5

    slot = h.controller.backends["vastai"]
    slot.paid_usd_per_h = 0.40
    slot.backend.balance_usd = balance  # type: ignore[attr-defined]
    state = h.controller.credit(slot, h.clock.now)
    assert state and state.left_usd == 7.5 and state.source == "platform"
    h.controller.credit(slot, h.clock.now + 60)
    assert len(calls) == 1
    h.controller.credit(slot, h.clock.now + 601)
    assert len(calls) == 2
