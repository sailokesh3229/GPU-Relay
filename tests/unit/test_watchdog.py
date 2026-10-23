from __future__ import annotations

from pathlib import Path

import pytest

from gpurelay.config import RuntimeSettings
from gpurelay.runtime.checkpoint import SnapshotInfo
from gpurelay.runtime.persist import PersistResult
from gpurelay.runtime.state import StateManifest
from gpurelay.runtime.watchdog import Watchdog


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


SETTINGS = RuntimeSettings(
    min_interval_s=120, max_lost_work_s=900, overhead_budget=0.03, safety_margin_s=300
)


def _snap(t_snap: float, size: int) -> SnapshotInfo:
    return SnapshotInfo(1, Path("."), t_snap, 0.0, size, StateManifest())


def test_interval_before_first_measurement_is_min() -> None:
    assert Watchdog(SETTINGS, None, Clock()).interval_s() == 120


@pytest.mark.parametrize(("t_snap", "expected"), [(1.0, 120), (9.0, 300), (60.0, 900)])
def test_interval_is_clamped(t_snap: float, expected: float) -> None:
    wd = Watchdog(SETTINGS, None, Clock())
    wd.record_snapshot(_snap(t_snap, 1))
    assert wd.interval_s() == pytest.approx(expected)


def test_checkpoint_due_after_interval() -> None:
    clock = Clock()
    wd = Watchdog(SETTINGS, None, clock)
    clock.now += 119
    assert not wd.checkpoint_due()
    clock.now += 2
    assert wd.checkpoint_due()


def test_handoff_condition_uses_snapshot_persist_gap_and_margin() -> None:
    clock = Clock()
    wd = Watchdog(SETTINGS, deadline=clock.now + 1000, clock=clock)
    wd.record_snapshot(_snap(t_snap=10.0, size=100_000_000))
    wd.record_persist(PersistResult(seq=1, ok=True, raw_bytes=100_000_000, total_s=50.0))
    wd.note_safe_point()
    clock.now += 40
    wd.note_safe_point()  # gap 40 s
    # needed = gap 40 + snapshot 10 + persist 50 + margin 300 = 400
    clock.now = wd.deadline - 401  # type: ignore[operator]
    assert not wd.handoff_due()
    clock.now = wd.deadline - 399  # type: ignore[operator]
    assert wd.handoff_due()


def test_no_deadline_never_hands_off() -> None:
    clock = Clock()
    wd = Watchdog(SETTINGS, None, clock)
    clock.now += 10**9
    assert not wd.handoff_due()


def test_signal_requests_handoff_and_second_signal_aborts() -> None:
    wd = Watchdog(SETTINGS, None, Clock())
    wd._on_signal(15, None)
    assert wd.handoff_due()
    with pytest.raises(KeyboardInterrupt):
        wd._on_signal(15, None)


def test_safe_point_gap_warning(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    wd = Watchdog(RuntimeSettings(max_lost_work_s=60), None, clock)
    wd.note_safe_point()
    clock.now += 61
    wd.note_safe_point()
    assert wd.max_gap_s == pytest.approx(61)
    assert "safe points are" in caplog.text
