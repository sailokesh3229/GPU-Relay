"""Relay one job through segments on a single local/simulated backend until it ends.

Used by ``gpurelay run-local`` and the equivalence harness. The full multi-backend
controller (scheduler, quota, parallel jobs) arrives in Phase 5.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from gpurelay.backends.local import LocalBackend, is_runtime_exit
from gpurelay.bundle.packager import upload_bundle
from gpurelay.controller.assemble import AssemblyResult, assemble
from gpurelay.models import ExitCode, Job, JobKind, JobState, Segment, StatusFile, new_id
from gpurelay.runtime.checkpoint import read_pointer
from gpurelay.storage.base import Storage
from gpurelay.storage.layout import status_key

log = logging.getLogger(__name__)


@dataclass
class SegmentRecord:
    segment_id: str
    exit_code: int | None
    outcome: str  # completed | handoff | crashed | failed_user | non_resumable | failed_infra
    duration_s: float
    committed_seq_after: int | None


@dataclass
class RunResult:
    job_id: str
    state: JobState
    segments: list[SegmentRecord] = field(default_factory=list)
    assembly: AssemblyResult | None = None
    message: str = ""


def _read_status(storage: Storage, job_id: str) -> StatusFile | None:
    raw = storage.get_bytes(status_key(job_id))
    return StatusFile.model_validate_json(raw) if raw else None


def _outcome(code: int | None) -> str:
    if code is None or not is_runtime_exit(code):
        return "crashed"
    return {
        ExitCode.COMPLETED: "completed",
        ExitCode.HANDOFF: "handoff",
        ExitCode.USER_ERROR: "failed_user",
        ExitCode.INFRA_ERROR: "failed_infra",
        ExitCode.NON_RESUMABLE: "non_resumable",
    }[ExitCode(code)]


def run_local_job(
    script: Path,
    *,
    storage: Storage,
    backend: LocalBackend,
    out_dir: Path,
    job_id: str | None = None,
    extra_files: Mapping[str, Path] | None = None,
    max_segments: int = 200,
    max_infra_retries: int = 3,
    max_no_progress: int = 3,
    poll_interval_s: float = 0.2,
    on_segment: Callable[[SegmentRecord], None] | None = None,
    on_console: Callable[[str], None] | None = None,
) -> RunResult:
    """Run ``script`` segment after segment until it completes or fails for good."""
    job = Job(
        id=job_id or new_id("job"),
        name=Path(script).stem,
        kind=JobKind.NOTEBOOK if Path(script).suffix == ".ipynb" else JobKind.SCRIPT,
        entry=Path(script).name,
    )
    upload_bundle(storage, job.id, Path(script), extra_files, include_wheel=False)
    result = RunResult(job_id=job.id, state=JobState.RUNNING)
    infra_failures = 0
    no_progress = 0

    for index in range(max_segments):
        before = read_pointer(storage, job.id)
        segment = Segment(job_id=job.id, index=index, backend=backend.name)
        started = time.monotonic()
        handle = backend.launch(job, segment, backend.max_session_s)
        console_pos = 0
        while backend.returncode(handle) is None:
            time.sleep(poll_interval_s)
            console_pos = _stream_console(handle.extra.get("console"), console_pos, on_console)
        code = backend.wait(handle)
        _stream_console(handle.extra.get("console"), console_pos, on_console)

        after = read_pointer(storage, job.id)
        outcome = _outcome(code)
        status = _read_status(storage, job.id)
        if outcome == "completed" and (status is None or status.state.value != "completed"):
            outcome = "crashed"  # exit 0 without a completed status is not trusted
        record = SegmentRecord(
            segment_id=segment.id,
            exit_code=code,
            outcome=outcome,
            duration_s=time.monotonic() - started,
            committed_seq_after=after.seq if after else None,
        )
        result.segments.append(record)
        if on_segment:
            on_segment(record)

        if outcome == "completed":
            result.assembly = assemble(storage, job.id, out_dir)
            result.state = JobState.COMPLETED
            return result
        if outcome == "failed_user":
            result.state = JobState.FAILED_USER
            result.message = status.message if status else "user code failed"
            return result
        if outcome == "non_resumable":
            result.state = JobState.FAILED_USER
            result.message = status.message if status else "non-resumable state"
            return result
        if outcome == "failed_infra":
            infra_failures += 1
            if infra_failures > max_infra_retries:
                result.state = JobState.FAILED_INFRA
                result.message = status.message if status else "infrastructure failure"
                return result

        progressed = (after.seq if after else None) != (before.seq if before else None)
        no_progress = 0 if progressed else no_progress + 1
        if no_progress >= max_no_progress:
            result.state = JobState.FAILED_INFRA
            result.message = (
                f"no checkpoint committed in {no_progress} consecutive segments; "
                "sessions are too short for the checkpoint interval"
            )
            return result

    result.state = JobState.FAILED_INFRA
    result.message = f"gave up after {max_segments} segments"
    return result


def _stream_console(path: str | None, pos: int, on_console: Callable[[str], None] | None) -> int:
    if on_console is None or not path:
        return pos
    try:
        with open(path, "rb") as fh:
            fh.seek(pos)
            data = fh.read()
    except FileNotFoundError:
        return pos
    if data:
        on_console(data.decode("utf-8", errors="replace"))
    return pos + len(data)
