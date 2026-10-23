"""Local backend (spec §10.2): runs the runtime in a subprocess on this machine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

from gpurelay.config import RuntimeSettings
from gpurelay.models import (
    BackendCaps,
    BackendRunState,
    ExitCode,
    Job,
    LaunchHandle,
    ReadyStatus,
    Segment,
)
from gpurelay.storage.base import Storage
from gpurelay.storage.factory import storage_uri

# Exit codes that mean "the runtime ended the session itself".
_RUNTIME_EXIT_CODES = frozenset(int(c) for c in ExitCode)


class LocalBackend:
    """Runs ``python -m gpurelay.runtime.entry`` in a subprocess per segment.

    Each segment gets a fresh working directory (like a new cloud session):
    ``<root>/<job>/<segment>/{work,runtime,console.txt}``.
    """

    automated = True

    def __init__(
        self,
        storage: Storage,
        root: Path,
        *,
        name: str = "local",
        settings: RuntimeSettings | None = None,
        max_session_s: float = 7 * 24 * 3600,
        gpu_label: str | None = None,
        python: str = sys.executable,
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.storage = storage
        self.root = Path(root)
        self.settings = settings or RuntimeSettings()
        self.max_session_s = max_session_s
        self.gpu_label = gpu_label
        self.python = python
        self.extra_env = dict(env or {})
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._consoles: dict[str, IO[bytes]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ Backend API

    def capabilities(self) -> BackendCaps:
        return BackendCaps(
            gpu_types=[self.gpu_label] if self.gpu_label else [],
            max_session_s=int(self.max_session_s) or 1,
            max_concurrent=4,
            needs_user_action=False,
        )

    def check_ready(self) -> ReadyStatus:
        return ReadyStatus(ready=True)

    def segment_dir(self, job_id: str, segment_id: str) -> Path:
        return self.root / job_id / segment_id

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        seg_dir = self.segment_dir(job.id, segment.id)
        seg_dir.mkdir(parents=True, exist_ok=True)
        console_path = seg_dir / "console.txt"
        deadline = time.time() + min(deadline_s, self.max_session_s)
        cmd = [
            self.python,
            "-m",
            "gpurelay.runtime.entry",
            "--job",
            job.id,
            "--segment",
            segment.id,
            "--storage",
            storage_uri(self.storage),
            "--work-dir",
            str(seg_dir / "work"),
            "--runtime-dir",
            str(seg_dir / "runtime"),
            "--backend",
            self.name,
            "--deadline",
            repr(deadline),
            "--settings",
            json.dumps(self.settings.model_dump(mode="json")),
        ]
        env = dict(os.environ)
        # Deterministic str hashing across segments (set/dict-of-str iteration order).
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MPLBACKEND", "Agg")
        env.update(self.extra_env)
        console = open(console_path, "wb")  # noqa: SIM115 - closed when the process ends
        proc = subprocess.Popen(
            cmd, stdout=console, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env
        )
        handle_id = f"{segment.id}:{proc.pid}"
        with self._lock:
            self._procs[handle_id] = proc
            self._consoles[handle_id] = console
        return LaunchHandle(
            backend=self.name,
            handle_id=handle_id,
            extra={"console": str(console_path), "pid": str(proc.pid)},
        )

    def poll(self, handle: LaunchHandle) -> BackendRunState:
        proc = self._procs.get(handle.handle_id)
        if proc is None:
            return BackendRunState.UNKNOWN
        code = proc.poll()
        if code is None:
            return BackendRunState.RUNNING
        self._close_console(handle.handle_id)
        if code in (ExitCode.COMPLETED, ExitCode.HANDOFF):
            return BackendRunState.FINISHED
        return BackendRunState.ERROR

    def returncode(self, handle: LaunchHandle) -> int | None:
        proc = self._procs.get(handle.handle_id)
        return None if proc is None else proc.poll()

    def wait(self, handle: LaunchHandle, timeout: float | None = None) -> int:
        proc = self._procs[handle.handle_id]
        code = proc.wait(timeout)
        self._close_console(handle.handle_id)
        return code

    def cancel(self, handle: LaunchHandle) -> None:
        proc = self._procs.get(handle.handle_id)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        self._close_console(handle.handle_id)

    def fetch_console_log(self, handle: LaunchHandle) -> str | None:
        path = handle.extra.get("console")
        if not path or not Path(path).exists():
            return None
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def _close_console(self, handle_id: str) -> None:
        with self._lock:
            console = self._consoles.pop(handle_id, None)
        if console is not None:
            console.close()


def is_runtime_exit(code: int | None) -> bool:
    """True if ``code`` was produced by the runtime itself (not a kill/crash)."""
    return code in _RUNTIME_EXIT_CODES
