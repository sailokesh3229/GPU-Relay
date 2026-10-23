"""Retry with exponential backoff for flaky I/O (storage uploads/downloads)."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from gpurelay.errors import StorageError, StorageNotFoundError

_T = TypeVar("_T")

log = logging.getLogger(__name__)

# Private generator: GPU Relay must never consume the user's global RNG streams.
_jitter_rng = random.Random()


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff: ``base_delay_s * multiplier**n`` capped at ``max_delay_s``.

    Only exceptions listed in ``retry_on`` (and not in ``give_up_on``) are retried;
    ``BaseException`` (e.g. ``KeyboardInterrupt``, simulated kills) always propagates
    immediately.
    """

    attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.1  # +/- fraction of the delay
    retry_on: tuple[type[Exception], ...] = (OSError, StorageError)
    give_up_on: tuple[type[Exception], ...] = (StorageNotFoundError, FileNotFoundError)
    sleep: Callable[[float], None] = field(default=time.sleep, compare=False, repr=False)

    def delay(self, attempt: int) -> float:
        """Delay before retry number ``attempt`` (0-based)."""
        base = min(self.max_delay_s, self.base_delay_s * self.multiplier**attempt)
        if self.jitter:
            base *= 1 + _jitter_rng.uniform(-self.jitter, self.jitter)
        return max(0.0, base)

    def call(self, fn: Callable[[], _T], *, what: str = "operation") -> _T:
        """Call ``fn`` until it succeeds or attempts are exhausted (then re-raise)."""
        for attempt in range(self.attempts):
            try:
                return fn()
            except self.retry_on as exc:
                if isinstance(exc, self.give_up_on) or attempt == self.attempts - 1:
                    raise
                wait = self.delay(attempt)
                log.warning(
                    "%s failed (%s: %s); retry %d/%d in %.2fs",
                    what,
                    type(exc).__name__,
                    exc,
                    attempt + 1,
                    self.attempts - 1,
                    wait,
                )
                self.sleep(wait)
        raise AssertionError("unreachable")


NO_RETRY = RetryPolicy(attempts=1)
