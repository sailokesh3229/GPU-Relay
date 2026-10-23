"""Simulated backend (spec §10.3): Local plus fault injection, for tests.

* hard kill (``SIGKILL`` / ``TerminateProcess``) after a fixed or random delay;
* graceful deadlines via a short ``max_session_s`` (forces handoffs);
* fake GPU-model labels.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gpurelay.backends.local import LocalBackend
from gpurelay.config import RuntimeSettings
from gpurelay.models import Job, LaunchHandle, Segment
from gpurelay.runtime.checkpoint import read_pointer
from gpurelay.storage.base import Storage


@dataclass
class KillEvent:
    """A simulated hard kill, with what was committed at that moment."""

    segment_id: str
    killed_at: float  # epoch seconds
    last_commit_created_at: datetime | None
    last_commit_segment: str | None


class SimulatedBackend(LocalBackend):
    """Local backend with simulated kills and deadlines."""

    def __init__(
        self,
        storage: Storage,
        root: Path,
        *,
        name: str = "simulated",
        settings: RuntimeSettings | None = None,
        max_session_s: float = 7 * 24 * 3600,
        gpu_label: str | None = "SimGPU",
        kill_after_s: float | None = None,
        kill_between_s: tuple[float, float] | None = None,
        seed: int = 0,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            storage,
            root,
            name=name,
            settings=settings,
            max_session_s=max_session_s,
            gpu_label=gpu_label,
            env=env,
        )
        if kill_after_s is not None and kill_between_s is not None:
            raise ValueError("use kill_after_s or kill_between_s, not both")
        self.kill_after_s = kill_after_s
        self.kill_between_s = kill_between_s
        self._rng = random.Random(seed)
        self.kills: list[KillEvent] = []
        self._timers: list[threading.Timer] = []

    def _kill_delay(self) -> float | None:
        if self.kill_after_s is not None:
            return self.kill_after_s
        if self.kill_between_s is not None:
            return self._rng.uniform(*self.kill_between_s)
        return None

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        handle = super().launch(job, segment, deadline_s)
        delay = self._kill_delay()
        if delay is not None:
            timer = threading.Timer(delay, self._hard_kill, args=(job.id, segment.id, handle))
            timer.daemon = True
            timer.start()
            self._timers.append(timer)
        return handle

    def _hard_kill(self, job_id: str, segment_id: str, handle: LaunchHandle) -> None:
        proc = self._procs.get(handle.handle_id)
        if proc is None or proc.poll() is not None:
            return
        proc.kill()
        killed_at = time.time()
        pointer = read_pointer(self.storage, job_id)
        self.kills.append(
            KillEvent(
                segment_id=segment_id,
                killed_at=killed_at,
                last_commit_created_at=pointer.created_at if pointer else None,
                last_commit_segment=pointer.segment_id if pointer else None,
            )
        )

    def cancel_timers(self) -> None:
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
