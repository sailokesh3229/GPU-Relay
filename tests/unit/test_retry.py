from __future__ import annotations

import random

import pytest

from fakes import SimulatedKill
from gpurelay.errors import StorageError, StorageNotFoundError
from gpurelay.retry import RetryPolicy


def _policy(sleeps: list[float], attempts: int = 4) -> RetryPolicy:
    return RetryPolicy(attempts=attempts, base_delay_s=1.0, jitter=0.0, sleep=sleeps.append)


def test_retries_then_succeeds_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    calls = iter([OSError("a"), StorageError("b"), "ok"])

    def flaky() -> str:
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    assert _policy(sleeps).call(flaky) == "ok"
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_attempts() -> None:
    sleeps: list[float] = []

    def dead() -> None:
        raise OSError("down")

    with pytest.raises(OSError, match="down"):
        _policy(sleeps, attempts=3).call(dead)
    assert len(sleeps) == 2


def test_delay_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=1, max_delay_s=5, jitter=0)
    assert [policy.delay(i) for i in range(5)] == [1, 2, 4, 5, 5]


@pytest.mark.parametrize("exc", [StorageNotFoundError("x"), ValueError("bad"), SimulatedKill()])
def test_not_retried(exc: BaseException) -> None:
    sleeps: list[float] = []

    def fail() -> None:
        raise exc

    with pytest.raises(type(exc)):
        _policy(sleeps).call(fail)
    assert sleeps == []


def test_jitter_does_not_touch_global_rng() -> None:
    """Retries must never shift the user's random stream (output equivalence)."""
    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    RetryPolicy(jitter=0.5).delay(3)
    assert random.random() == expected
