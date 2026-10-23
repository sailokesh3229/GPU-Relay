"""Deadline watchdog and adaptive checkpoint policy (spec §8.2, §12.4).

* ``interval = clamp(t_snap / overhead_budget, min_interval_s, max_lost_work_s)``;
  a checkpoint is taken at the first safe point after ``interval`` since the last one.
* **Handoff** (forced checkpoint, then exit 85) when
  ``time_to_deadline < gap + t_snap + S/bw + safety_margin`` or after SIGTERM/SIGINT.
  ``gap`` is the last observed time between safe points: the check runs at a safe point,
  and the *next* one may already be too late (an addition to the spec's formula).
* Safe-point gaps longer than ``max_lost_work_s`` are warned about once.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

from gpurelay.config import RuntimeSettings
from gpurelay.runtime.checkpoint import SnapshotInfo
from gpurelay.runtime.persist import PersistResult

log = logging.getLogger(__name__)

_SIGNALS = tuple(
    sig for name in ("SIGTERM", "SIGINT") if (sig := getattr(signal, name, None)) is not None
)


class Watchdog:
    """Decides when to checkpoint and when to hand off."""

    def __init__(
        self,
        settings: RuntimeSettings,
        deadline: float | None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.deadline = deadline
        self.clock = clock
        self.t_snap_s: float | None = None
        self.size_bytes: int | None = None
        self.throughput_bps: float | None = None
        self.last_checkpoint_at = clock()
        self.stop_requested = False
        self._last_safe_point_at: float | None = None
        self.last_gap_s = 0.0
        self.max_gap_s = 0.0
        self._gap_warned = False
        self._saved_handlers: dict[int, Any] = {}

    # ------------------------------------------------------------------ measurements

    def reset_timer(self) -> None:
        """Start counting the checkpoint interval from now (segment start / resume point)."""
        self.last_checkpoint_at = self.clock()
        self._last_safe_point_at = None

    def record_snapshot(self, info: SnapshotInfo) -> None:
        self.t_snap_s = info.t_snap_s
        self.size_bytes = info.size_bytes
        self.last_checkpoint_at = self.clock()

    def record_persist(self, result: PersistResult | None) -> None:
        if result is not None and result.ok and result.throughput_bps > 0:
            self.throughput_bps = result.throughput_bps

    # ------------------------------------------------------------------ policy

    def interval_s(self) -> float:
        s = self.settings
        if self.t_snap_s is None:
            return s.min_interval_s
        wanted = self.t_snap_s / s.overhead_budget
        return min(max(wanted, s.min_interval_s), s.max_lost_work_s)

    def estimated_checkpoint_s(self) -> float:
        """Time a checkpoint takes end to end: snapshot + persist."""
        snap = self.t_snap_s or 0.0
        persist = 0.0
        if self.size_bytes and self.throughput_bps:
            persist = self.size_bytes / self.throughput_bps
        return snap + persist

    def time_left_s(self) -> float | None:
        return None if self.deadline is None else self.deadline - self.clock()

    def note_safe_point(self) -> None:
        now = self.clock()
        if self._last_safe_point_at is not None:
            gap = now - self._last_safe_point_at
            self.last_gap_s = gap
            self.max_gap_s = max(self.max_gap_s, gap)
            if gap > self.settings.max_lost_work_s and not self._gap_warned:
                self._gap_warned = True
                log.warning(
                    "safe points are %.0fs apart, more than max_lost_work_s=%.0fs; an "
                    "unexpected kill may lose more work. Consider relay.range/relay.chunks "
                    "or relay.safe_point() inside the long iteration.",
                    gap,
                    self.settings.max_lost_work_s,
                )
        self._last_safe_point_at = now

    def checkpoint_due(self) -> bool:
        return self.clock() - self.last_checkpoint_at >= self.interval_s()

    def handoff_due(self, lookahead_s: float | None = None) -> bool:
        """Hand off now?

        ``lookahead_s`` is the expected time until the next acceptable safe point; when
        omitted, the last observed gap between safe points is used. Framework hooks pass
        their own estimate (step time x steps to the next aligned save + save time).
        """
        if self.stop_requested:
            return True
        left = self.time_left_s()
        if left is None:
            return False
        gap = self.last_gap_s if lookahead_s is None else lookahead_s
        return left < gap + self.estimated_checkpoint_s() + self.settings.safety_margin_s

    # ------------------------------------------------------------------ signals

    def install_signal_handlers(self) -> None:
        """SIGTERM/SIGINT request a handoff at the next safe point; a second one aborts."""
        if not _is_main_thread():
            return
        for sig in _SIGNALS:
            self._saved_handlers[sig] = signal.signal(sig, self._on_signal)

    def uninstall_signal_handlers(self) -> None:
        for sig, handler in self._saved_handlers.items():
            signal.signal(sig, handler)
        self._saved_handlers.clear()

    def _on_signal(self, signum: int, _frame: FrameType | None) -> None:
        if self.stop_requested:
            raise KeyboardInterrupt(f"second signal {signum}: aborting without checkpoint")
        self.stop_requested = True
        log.warning(
            "signal %d received: will checkpoint and hand off at the next safe point", signum
        )


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()
