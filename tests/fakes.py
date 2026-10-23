"""Test doubles shared by unit and integration tests."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from gpurelay.storage import LocalFSStorage


class FakeBackend:
    """In-memory backend for controller tests: sessions end when the test says so.

    ``finish(segment_id, code, state)`` writes the runtime's ``status.json`` the way a real
    session would and makes ``poll`` report the session as ended.
    """

    automated = True

    def __init__(self, name: str, storage: Any, *, automated: bool = True, ready: bool = True):
        from gpurelay.models import BackendCaps

        self.name = name
        self.storage = storage
        self.automated = automated
        self.ready = ready
        self.caps = BackendCaps(gpu_types=[], max_session_s=3600, max_concurrent=4)
        self.launched: list[tuple[str, str, float]] = []  # (job id, segment id, deadline_s)
        self.cancelled: list[str] = []
        self._codes: dict[str, int | None] = {}

    def capabilities(self) -> Any:
        return self.caps

    def check_ready(self) -> Any:
        from gpurelay.models import ReadyStatus

        return ReadyStatus(ready=self.ready)

    def launch(self, job: Any, segment: Any, deadline_s: float) -> Any:
        from gpurelay.models import LaunchHandle

        self.launched.append((job.id, segment.id, deadline_s))
        self._codes[segment.id] = None
        return LaunchHandle(backend=self.name, handle_id=segment.id, extra={"job": job.id})

    def poll(self, handle: Any) -> Any:
        from gpurelay.models import BackendRunState

        if handle.handle_id not in self._codes or not self.automated:
            return BackendRunState.UNKNOWN  # manual backends cannot see the session (Colab)
        code = self._codes[handle.handle_id]
        if code is None:
            return BackendRunState.RUNNING
        return BackendRunState.FINISHED if code in (0, 85) else BackendRunState.ERROR

    def returncode(self, handle: Any) -> int | None:
        return self._codes.get(handle.handle_id)

    def cancel(self, handle: Any) -> None:
        self.cancelled.append(handle.handle_id)
        self._codes[handle.handle_id] = -9

    def fetch_console_log(self, handle: Any) -> str | None:
        return None

    def finish(
        self, job_id: str, segment_id: str, code: int, state: str, message: str = ""
    ) -> None:
        from gpurelay.models import RuntimeState, StatusFile, utcnow
        from gpurelay.storage.layout import status_key

        status = StatusFile(
            job_id=job_id,
            segment_id=segment_id,
            state=RuntimeState(state),
            heartbeat_at=utcnow(),
            message=message,
        )
        self.storage.put_bytes(status.model_dump_json().encode(), status_key(job_id))
        self._codes[segment_id] = code

    def running(self) -> list[tuple[str, str]]:
        return [(j, s) for j, s, _d in self.launched if self._codes.get(s) is None]


class SimulatedKill(BaseException):
    """Raised to simulate the process dying at a precise point (not an Exception,
    so retry logic and ``except Exception`` handlers do not swallow it)."""


class FaultyStorage:
    """Wraps a LocalFSStorage and injects faults into ``put_file``/``put_bytes``.

    Deliberately does **not** expose ``sha256`` so upload verification takes the
    download-and-hash path.

    * ``fail_puts``: the next N puts raise ``OSError`` (transient outage).
    * ``corrupt_puts``: the next N puts store corrupted bytes.
    * ``kill_after_puts``: after N more successful puts, the next put raises SimulatedKill.
    * ``exit_after_puts``: after N more successful puts, the next put calls ``os._exit``.
    * ``put_delay_s``: sleep inside every put (to observe concurrency).
    """

    def __init__(self, inner: LocalFSStorage) -> None:
        self.inner = inner
        self.fail_puts = 0
        self.corrupt_puts = 0
        self.kill_after_puts: int | None = None
        self.exit_after_puts: int | None = None
        self.put_delay_s = 0.0
        self.puts = 0
        self.active_puts = 0
        self.max_active_puts = 0
        self._lock = threading.Lock()

    def _before_put(self) -> bool:
        """Apply faults; return True if this put must be corrupted."""
        if self.exit_after_puts is not None:
            if self.exit_after_puts == 0:
                os._exit(9)
            self.exit_after_puts -= 1
        if self.kill_after_puts is not None:
            if self.kill_after_puts == 0:
                self.kill_after_puts = None
                raise SimulatedKill("killed during persist")
            self.kill_after_puts -= 1
        if self.fail_puts > 0:
            self.fail_puts -= 1
            raise OSError("simulated storage outage")
        if self.corrupt_puts > 0:
            self.corrupt_puts -= 1
            return True
        return False

    def _tracked(self, write: object) -> None:
        with self._lock:
            self.active_puts += 1
            self.max_active_puts = max(self.max_active_puts, self.active_puts)
        try:
            if self.put_delay_s:
                time.sleep(self.put_delay_s)
            write()  # type: ignore[operator]
            self.puts += 1
        finally:
            with self._lock:
                self.active_puts -= 1

    def put_file(self, local: Path, remote: str) -> None:
        if self._before_put():
            self._tracked(lambda: self.inner.put_bytes(b"corrupted", remote))
        else:
            self._tracked(lambda: self.inner.put_file(local, remote))

    def put_bytes(self, data: bytes, remote: str) -> None:
        corrupt = self._before_put()
        self._tracked(lambda: self.inner.put_bytes(b"corrupted" if corrupt else data, remote))

    def get_file(self, remote: str, local: Path) -> None:
        self.inner.get_file(remote, local)

    def get_bytes(self, remote: str) -> bytes | None:
        return self.inner.get_bytes(remote)

    def list(self, prefix: str) -> list[str]:
        return self.inner.list(prefix)

    def delete(self, remote: str) -> None:
        self.inner.delete(remote)

    def exists(self, remote: str) -> bool:
        return self.inner.exists(remote)
