"""Outside GPU Relay the SDK must behave like plain Python (spec §7.4)."""

from __future__ import annotations

import pytest

import gpurelay as relay


def test_not_active() -> None:
    assert relay.is_active() is False


def test_range_is_builtin_range() -> None:
    assert list(relay.range(5)) == [0, 1, 2, 3, 4]
    assert list(relay.range(2, 10, 3)) == [2, 5, 8]


def test_state_returns_view() -> None:
    history: list[float] = []
    state = relay.state(history=history, best=1.0)
    assert state.history is history
    state.best = 0.5
    assert state["best"] == 0.5
    assert "history" in state
    with pytest.raises(AttributeError):
        _ = state.missing


def test_chunks_and_results_from_sequence() -> None:
    items = list(range(23))
    seen = []
    for chunk in relay.chunks(items, size=5):
        seen.append((chunk.id, chunk.start, list(chunk)))
        relay.save_result(chunk.id, sum(chunk.items))
    assert [s[0] for s in seen] == [0, 1, 2, 3, 4]
    assert seen[-1] == (4, 20, [20, 21, 22])
    assert relay.load_results() == [sum(items[i : i + 5]) for i in range(0, 23, 5)]


def test_chunks_from_iterator_and_named_results() -> None:
    for chunk in relay.chunks((x * x for x in range(7)), size=3, name="squares"):
        relay.save_result(chunk.id, list(chunk))
    for chunk in relay.chunks([1, 2], size=1, name="other"):
        relay.save_result(chunk.id, chunk.items[0])
    assert relay.load_results("squares") == [[0, 1, 4], [9, 16, 25], [36]]
    assert relay.load_results() == [1, 2]


def test_save_result_outside_chunks_fails() -> None:
    with pytest.raises(RuntimeError, match=r"inside a relay.chunks"):
        relay.save_result(0, 1)


def test_chunk_size_validated() -> None:
    with pytest.raises(ValueError):
        relay.chunks([1], size=0)


def test_safe_point_is_noop() -> None:
    relay.safe_point()
