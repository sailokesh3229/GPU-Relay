"""``gpurelay analyze --probe``: run the job here for a short time and measure it (spec §15).

The job runs under the real runtime (same entry point as every backend) against a throwaway
local store, with a deadline: it hands off at the first safe point after ``seconds``. From the
run we report startup time (until the first progress report), the outermost loop's rate and
the projected total on this machine, peak memory of the process tree, and the size of the
handoff checkpoint. Nothing is uploaded and nothing is kept.

Numbers are for *this* computer (often CPU only); a GPU session is usually faster.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpurelay.bundle.packager import upload_bundle
from gpurelay.config import RuntimeSettings
from gpurelay.models import ExitCode, StatusFile
from gpurelay.runtime.checkpoint import read_pointer
from gpurelay.storage.layout import ckpt_prefix, job_prefix, segment_env_key, status_key
from gpurelay.storage.localfs import LocalFSStorage

PROBE_JOB = "job-probe"
PROBE_SEGMENT = "seg-probe"
GRACE_S = 120.0  # after the deadline: time to reach a safe point and write the checkpoint
SAMPLE_S = 0.5


@dataclass
class ProbeResult:
    seconds: float  # wall time of the run
    ended: str  # "handed off" | "completed" | "failed (exit N)" | "stopped (no safe point)"
    startup_s: float | None  # until the runtime first reported loop progress
    completed: int | None  # outermost loop iterations done
    total: int | None
    rate_per_h: float | None
    projected_h: float | None  # startup + total / rate, on this machine
    peak_rss_mb: float | None
    checkpoint_mb: float | None
    device: str
    log_tail: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _rss_mb(pid: int) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
        total = 0
        for proc in procs:
            try:
                total += proc.memory_info().rss
            except psutil.Error:
                continue
        return total / 2**20
    except psutil.Error:
        return None


def _status(storage: LocalFSStorage) -> StatusFile | None:
    raw = storage.get_bytes(status_key(PROBE_JOB))
    try:
        return StatusFile.model_validate_json(raw) if raw else None
    except ValueError:
        return None


def _checkpoint_mb(storage: LocalFSStorage) -> float | None:
    pointer = read_pointer(storage, PROBE_JOB)
    if pointer is None:
        return None
    total = 0
    for prefix in (ckpt_prefix(PROBE_JOB, pointer.seq), f"{job_prefix(PROBE_JOB)}/blobs"):
        for key in storage.list(prefix + "/"):
            total += storage.root.joinpath(*key.split("/")).stat().st_size
    return total / 2**20


def _device(storage: LocalFSStorage) -> str:
    raw = storage.get_bytes(segment_env_key(PROBE_JOB, PROBE_SEGMENT))
    if not raw:
        return "unknown"
    gpus = json.loads(raw).get("gpus") or []  # device names (torch), when the job uses CUDA
    return ", ".join(str(g) for g in gpus) or "CPU"


def probe(
    path: Path,
    *,
    seconds: float = 60.0,
    requirements: Path | None = None,
    env: dict[str, str] | None = None,
) -> ProbeResult:
    """Run ``path`` for about ``seconds`` under the runtime and measure it."""
    with tempfile.TemporaryDirectory(prefix="gpurelay-probe-") as tmp_name:
        tmp = Path(tmp_name)
        storage = LocalFSStorage(tmp / "store")
        upload_bundle(storage, PROBE_JOB, path, requirements=requirements, include_wheel=False)
        settings = RuntimeSettings(heartbeat_s=1, safety_margin_s=min(5.0, seconds / 6))
        command = [
            sys.executable, "-m", "gpurelay.runtime.entry",
            "--job", PROBE_JOB, "--segment", PROBE_SEGMENT,
            "--storage", f"localfs:{storage.root}",
            "--work-dir", str(tmp / "work"), "--runtime-dir", str(tmp / "runtime"),
            "--backend", "probe",
            "--deadline", repr(time.time() + seconds),
            "--settings", settings.model_dump_json(),
        ]  # fmt: skip
        child_env = dict(os.environ, **(env or {}), PYTHONHASHSEED="0", MPLBACKEND="Agg")
        log_path = tmp / "console.txt"
        started = time.monotonic()
        startup = peak = None
        live: StatusFile | None = None  # last status sampled while the loop was running
        with open(log_path, "wb") as console:
            proc = subprocess.Popen(command, env=child_env, stdout=console, stderr=console)
            next_status = 0.0
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > seconds + GRACE_S:
                    proc.kill()
                    break
                rss = _rss_mb(proc.pid)
                if rss is not None:
                    peak = max(peak or 0.0, rss)
                if elapsed >= next_status:
                    next_status = elapsed + 1.0
                    status = _status(storage)
                    if status is not None and status.progress.loop:
                        startup = elapsed if startup is None else startup
                        live = status
                time.sleep(SAMPLE_S)
            code = proc.wait()
        wall = time.monotonic() - started
        completed = total = None
        if live is not None:
            completed, total = next(iter(live.progress.loop.values()))
        rate = live.rate.per_hour if live is not None and live.rate else None
        projected = None
        if rate and total is not None:
            projected = (startup or 0.0) / 3600 + total / rate
        if code == ExitCode.HANDOFF:
            ended = "handed off"
        elif code == ExitCode.COMPLETED:
            ended = "completed"
            if total is None or projected is None:
                projected = wall / 3600  # the whole job fit in the probe
        elif wall > seconds + GRACE_S:
            ended = "stopped (no safe point reached after the deadline)"
        else:
            ended = f"failed (exit {code})"
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        return ProbeResult(
            seconds=wall,
            ended=ended,
            startup_s=startup,
            completed=completed,
            total=total,
            rate_per_h=rate,
            projected_h=projected,
            peak_rss_mb=peak,
            checkpoint_mb=_checkpoint_mb(storage),
            device=_device(storage),
            log_tail="\n".join(log_text.splitlines()[-15:]),
        )
