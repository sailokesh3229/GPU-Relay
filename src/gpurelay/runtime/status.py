"""``status.json`` heartbeat (spec §12.2): the source of truth for job progress."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from gpurelay.models import ParallelStatus, Progress, Rate, RuntimeState, StatusFile, utcnow
from gpurelay.retry import RetryPolicy
from gpurelay.storage.base import Storage
from gpurelay.storage.layout import status_key

log = logging.getLogger(__name__)

_STATUS_RETRY = RetryPolicy(attempts=3, base_delay_s=0.5)


@dataclass
class StatusSnapshot:
    """Live values the heartbeat publishes."""

    progress: Progress = field(default_factory=Progress)
    rate: Rate | None = None
    last_checkpoint_seq: int | None = None
    message: str = ""
    parallel: ParallelStatus | None = None


class StatusWriter:
    """Writes ``status.json`` on state changes and every ``interval_s`` (heartbeat)."""

    def __init__(
        self,
        storage: Storage,
        job_id: str,
        segment_id: str,
        interval_s: float,
        snapshot: Callable[[], StatusSnapshot],
        on_tick: Callable[[], None] | None = None,
        key: str | None = None,
    ) -> None:
        self.storage = storage
        self.key = key or status_key(job_id)  # workers write their own heartbeat file
        self.job_id = job_id
        self.segment_id = segment_id
        self.interval_s = interval_s
        self._snapshot = snapshot
        self._on_tick = on_tick
        self.state = RuntimeState.RUNNING
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def write(self, state: RuntimeState | None = None, message: str | None = None) -> None:
        """Write status now. Failures are logged, never raised."""
        with self._lock:
            if state is not None:
                self.state = state
            snap = self._snapshot()
            status = StatusFile(
                job_id=self.job_id,
                segment_id=self.segment_id,
                state=self.state,
                heartbeat_at=utcnow(),
                last_checkpoint_seq=snap.last_checkpoint_seq,
                progress=snap.progress,
                rate=snap.rate,
                message=message if message is not None else snap.message,
                parallel=snap.parallel,
            )
            body = status.model_dump_json().encode()
            try:
                _STATUS_RETRY.call(
                    lambda: self.storage.put_bytes(body, self.key),
                    what="write status.json",
                )
            except Exception as exc:
                log.warning("could not write status.json: %s", exc)

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(target=self._run, name="gpurelay-heartbeat", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.write()
            if self._on_tick is not None:
                try:
                    self._on_tick()
                except Exception as exc:  # never let a control poll kill the heartbeat
                    log.warning("control poll failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
