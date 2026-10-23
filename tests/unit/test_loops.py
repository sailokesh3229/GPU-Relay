"""Position tracking and replay-to-resume-point (runtime/loops.py)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from gpurelay.runtime.loops import Position, ResumeMismatchError, Tracker, run_loop

SafePoint = Callable[[Position], None]


def _loop(tracker: Tracker, n: int, on_sp: SafePoint) -> Iterator[int]:
    return run_loop(tracker, n, None, lambda s: ((i, i) for i in range(s, n)), on_sp)


def _explicit_sp(tracker: Tracker, on_sp: SafePoint) -> None:
    position = tracker.safe_point()
    if position is not None:
        on_sp(position)


def program(tracker: Tracker, out: list[Any], on_sp: SafePoint) -> None:
    """Nested loops, an explicit safe point, a loop that breaks, and a final loop."""
    out.append("start")
    for a in _loop(tracker, 3, on_sp):
        out.append(("A", a))
        for b in _loop(tracker, 2, on_sp):
            out.append(("B", a, b))
        _explicit_sp(tracker, on_sp)
        out.append(("A-end", a))
    _explicit_sp(tracker, on_sp)
    out.append("mid")
    for c in _loop(tracker, 5, on_sp):
        if c == 3:
            break
        out.append(("C", c))
    for d in _loop(tracker, 2, on_sp):
        out.append(("D", d))
    tracker.end_statement()
    out.append("end")


def _tracker(target: Position | None = None, on_reach: Any = lambda: None) -> Tracker:
    tracker = Tracker(on_reach=on_reach)
    tracker.begin_statement(0, target=target)
    return tracker


def _record_run() -> tuple[list[Any], list[tuple[Position, int]]]:
    out: list[Any] = []
    points: list[tuple[Position, int]] = []
    tracker = _tracker()
    program(tracker, out, lambda pos: points.append((pos, len(out))))
    return out, points


def test_live_run_positions_are_unique() -> None:
    out, points = _record_run()
    assert out[-1] == "end"
    keys = [repr(p) for p, _ in points]
    assert len(keys) == len(set(keys))
    # 3 A-iterations x (2 B-ends + 1 explicit + 1 A-end) + 1 explicit + 3 C + 2 D
    assert len(points) == 3 * 4 + 1 + 3 + 2


def test_resume_from_every_safe_point_reproduces_the_run() -> None:
    full, points = _record_run()
    for position, out_len in points:
        resumed: list[Any] = []
        reach_at: list[int] = []
        tracker = _tracker(position, lambda r=resumed, a=reach_at: a.append(len(r)))
        program(tracker, resumed, lambda _p: None)
        assert reach_at, f"resume point {position} never reached"
        combined = full[:out_len] + resumed[reach_at[0] :]
        assert combined == full, f"resume from {position}"


def test_finished_loops_are_skipped_not_replayed() -> None:
    _full, points = _record_run()
    position = next(
        p for p, _ in points if p["path"] and p["path"][0][0] == 2
    )  # in loop C (event 2)
    resumed: list[Any] = []
    reach_at: list[int] = []
    tracker = _tracker(position, lambda: reach_at.append(len(resumed)))
    program(tracker, resumed, lambda _p: None)
    replayed = resumed[: reach_at[0]]
    assert not any(isinstance(x, tuple) and x[0] in ("A", "B") for x in replayed)


def test_resume_into_nested_iteration_replays_only_that_iteration() -> None:
    _full, points = _record_run()
    # End of B iteration 1 inside A iteration 2.
    position = {"stmt": 0, "path": [[0, 2, None], [0, 1, None]], "end": "iter"}
    assert any(p == position for p, _ in points)
    resumed: list[Any] = []
    reach_at: list[int] = []
    tracker = _tracker(position, lambda: reach_at.append(len(resumed)))
    program(tracker, resumed, lambda _p: None)
    assert resumed[: reach_at[0]] == ["start", ("A", 2)]


@pytest.mark.parametrize(
    "target",
    [
        {"stmt": 0, "path": [[0, 9]], "end": "iter"},  # more iterations than the loop has
        {"stmt": 0, "path": [[7, 0]], "end": "iter"},  # loop that never starts
        {"stmt": 0, "path": [], "end": "sp", "sp": 9},  # safe point that never happens
    ],
)
def test_divergence_is_detected(target: Position) -> None:
    tracker = _tracker(target)
    with pytest.raises(ResumeMismatchError):
        program(tracker, [], lambda _p: None)


def test_mismatch_cannot_be_swallowed_by_user_code() -> None:
    assert not issubclass(ResumeMismatchError, Exception)


def test_progress_reports_active_loops() -> None:
    tracker = _tracker()
    seen: list[dict[str, tuple[int, int | None]]] = []
    for _ in _loop(tracker, 3, lambda _p: seen.append(tracker.progress())):
        pass
    assert seen[-1] == {"L0": (3, 3)}
    assert tracker.progress() == {}


def test_passthrough_runs_loops_fully_without_safe_points() -> None:
    out: list[Any] = []
    tracker = Tracker(on_reach=lambda: None)
    tracker.begin_statement(0, passthrough=True)
    points: list[Position] = []
    program(tracker, out, points.append)
    assert points == []
    assert out == _record_run()[0]


def test_loop_id_mismatch_is_detected() -> None:
    tracker = Tracker(on_reach=lambda: None)
    tracker.begin_statement(3, target={"stmt": 3, "path": [[0, 1, "L5-aaaaaa"]], "end": "iter"})
    with pytest.raises(ResumeMismatchError, match="L9-bbbbbb"):
        list(
            run_loop(
                tracker,
                4,
                None,
                lambda s: ((i, i) for i in range(s, 4)),
                lambda _p: None,
                loop_id="L9-bbbbbb",
            )
        )


def test_positions_carry_statement_index() -> None:
    tracker = Tracker(on_reach=lambda: None)
    tracker.begin_statement(7)
    points: list[Position] = []
    for _ in _loop(tracker, 1, points.append):
        pass
    assert points == [{"stmt": 7, "path": [[0, 0, None]], "end": "iter"}]


def test_adaptive_nested_loop_stops_tracking_tiny_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gpurelay.runtime.loops as loops_mod

    clock = [0.0]
    monkeypatch.setattr(loops_mod.time, "perf_counter", lambda: clock[0])
    tracker = _tracker()
    tracker.begin_statement(0)
    points: list[Position] = []

    def nested(step_s: float) -> list[Any]:
        points.clear()
        out: list[Any] = []
        for outer in run_loop(tracker, 1, None, lambda s: ((i, i) for i in range(s, 1)),
                              points.append):  # fmt: skip
            inner = run_loop(tracker, 100, None, lambda s: ((i, i) for i in range(s, 100)),
                             points.append, adaptive=True)  # fmt: skip
            for item in inner:
                clock[0] += step_s
                out.append((outer, item))
                assert tracker.loop_start(None).passthrough == (tracker.untracked > 0)
        return out

    fast = nested(1e-6)  # 1 us per iteration: untracked after the probe
    assert len(fast) == 100 and tracker.untracked == 0
    assert len([p for p in points if len(p["path"]) == 2]) == loops_mod.PROBE_ITERATIONS
    slow = nested(0.01)  # 10 ms per iteration: tracked to the end
    assert len(slow) == 100
    assert len([p for p in points if len(p["path"]) == 2]) == 100


def test_sparse_mode_offers_safe_points_every_stride(monkeypatch: pytest.MonkeyPatch) -> None:
    import gpurelay.runtime.loops as loops_mod

    clock = [0.0]
    monkeypatch.setattr(loops_mod.time, "perf_counter", lambda: clock[0])
    tracker = _tracker()
    tracker.begin_statement(0)
    points: list[Position] = []
    n = 2000
    seen = []
    for item in run_loop(tracker, n, None, lambda s: ((i, i) for i in range(s, n)),
                         points.append, adaptive=True):  # fmt: skip
        clock[0] += 1e-4  # 0.1 ms per iteration: sparse, stride = 0.05 s / 0.1 ms = 500
        seen.append(item)
        assert tracker.loop_start(None).passthrough == (len(seen) > loops_mod.PROBE_ITERATIONS)
    assert seen == list(range(n)) and tracker.untracked == 0
    sparse = [p["path"][-1][1] for p in points[loops_mod.PROBE_ITERATIONS :]]
    assert sparse == [531, 1031, 1531]  # iteration ends offered after the probe
    # Each sparse position resumes like any iteration end.
    resumed = _tracker(target=points[-1])
    out = list(run_loop(resumed, n, None, lambda s: ((i, i) for i in range(s, n)), points.append))
    assert out == list(range(1532, n))
