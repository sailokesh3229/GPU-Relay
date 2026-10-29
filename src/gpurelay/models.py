"""Core data models and enums shared by the controller, backends, and runtime.

These are plain pydantic models; persistence (SQLite, storage JSON) is handled by
the modules that own each record.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Return a short unique id such as ``job-1a2b3c4d5e6f``."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ====================================================================== enums


class JobState(str, Enum):
    """Job lifecycle (spec §13.5)."""

    QUEUED = "queued"
    LAUNCHING = "launching"
    RUNNING = "running"
    HANDOFF = "handoff"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    AWAITING_USER = "awaiting_user"
    WAITING_FOR_QUOTA = "waiting_for_quota"
    PAUSED = "paused"
    FAILED_USER = "failed_user"
    FAILED_INFRA = "failed_infra"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True when the job will never run again without user intervention."""
        return self in _TERMINAL_JOB_STATES


_TERMINAL_JOB_STATES = frozenset(
    {JobState.COMPLETED, JobState.FAILED_USER, JobState.FAILED_INFRA, JobState.CANCELLED}
)


class SegmentState(str, Enum):
    """State of one contiguous execution of a job on one backend session."""

    LAUNCHING = "launching"
    AWAITING_USER = "awaiting_user"  # Colab: notebook generated, user must press Run
    RUNNING = "running"
    HANDOFF = "handoff"  # planned stop after a checkpoint
    COMPLETED = "completed"
    CRASHED = "crashed"  # unplanned stop; resume from last committed checkpoint
    FAILED_USER = "failed_user"
    FAILED_INFRA = "failed_infra"
    CANCELLED = "cancelled"


class RuntimeState(str, Enum):
    """``state`` field of ``status.json`` written by the runtime (spec §12.2)."""

    RUNNING = "running"
    HANDOFF = "handoff"
    COMPLETED = "completed"
    FAILED_USER = "failed_user"
    FAILED_INFRA = "failed_infra"


class ExitCode(IntEnum):
    """Runtime process exit codes (spec §12.3)."""

    COMPLETED = 0
    HANDOFF = 85
    USER_ERROR = 86  # deterministic failure in user code; never retried
    INFRA_ERROR = 87  # infra/runtime failure; retry elsewhere
    NON_RESUMABLE = 88  # non-resumable state detected and not accepted


class BackendRunState(str, Enum):
    """Backend-reported session state, as returned by ``Backend.poll`` (spec §10.1)."""

    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    UNKNOWN = "unknown"


class ResumabilityLevel(str, Enum):
    """Analyzer verdict for a block, cell, or whole job (spec §14)."""

    FULL = "FULL"
    PARTIAL = "PARTIAL"
    RESTART_ONLY = "RESTART_ONLY"


class BudgetModel(str, Enum):
    """How a backend's GPU budget is replenished (spec §13.2)."""

    WEEKLY_ROLLING = "weekly_rolling"
    MONTHLY = "monthly"
    DAILY = "daily"  # resets at 00:00 UTC
    UNKNOWN_OPPORTUNISTIC = "unknown_opportunistic"


class JobKind(str, Enum):
    SCRIPT = "script"
    NOTEBOOK = "notebook"


class DataSourceKind(str, Enum):
    """Prefixes accepted by ``--data SPEC`` (spec §15)."""

    KAGGLE = "kaggle"
    HF = "hf"
    URL = "url"
    LOCAL = "local"


# ====================================================================== records


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSpec(_Model):
    """A data source attached to a job, e.g. ``kaggle:owner/dataset``."""

    kind: DataSourceKind
    ref: str = Field(min_length=1)

    @classmethod
    def parse(cls, spec: str) -> DataSpec:
        """Parse a ``--data`` CLI value of the form ``<kind>:<ref>``."""
        kind, sep, ref = spec.partition(":")
        if not sep or not ref:
            raise ValueError(f"invalid data spec {spec!r}: expected '<kind>:<ref>'")
        try:
            source = DataSourceKind(kind)
        except ValueError:
            allowed = ", ".join(k.value for k in DataSourceKind)
            raise ValueError(f"invalid data spec {spec!r}: kind must be one of {allowed}") from None
        if source is DataSourceKind.URL and not ref.startswith(("https://", "http://")):
            raise ValueError(f"invalid data spec {spec!r}: url must start with http(s)://")
        if source in (DataSourceKind.KAGGLE, DataSourceKind.HF) and ref.count("/") != 1:
            raise ValueError(f"invalid data spec {spec!r}: expected '<owner>/<name>'")
        return cls(kind=source, ref=ref)

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.ref}"


class Job(_Model):
    """One user workload submitted to GPU Relay."""

    id: str = Field(default_factory=lambda: new_id("job"))
    name: str
    kind: JobKind
    entry: str  # path of the script/notebook inside the bundle
    state: JobState = JobState.QUEUED
    priority: int = 0
    submitted_at: datetime = Field(default_factory=utcnow)
    requirements: str | None = None
    data: list[DataSpec] = Field(default_factory=list)
    estimate_h: float | None = Field(default=None, gt=0)
    # Paid GPUs this job may use once free ones run out, in USD (default: free only).
    max_paid_usd: float = Field(default=0.0, ge=0)
    prefer_backend: str | None = None
    accept_restart_only: bool = False
    bundle_hash: str | None = None
    # A rerun with an AI fix (spec §20): the attempt it replaces and the fix it applies.
    retry_of: str | None = None
    fix_id: str | None = None
    # Controller bookkeeping (spec §13.5).
    infra_failures: int = 0
    message: str = ""
    requested: str | None = None  # "pause" | "resume" | "cancel" (set by the CLI)
    wait_until: datetime | None = None  # waiting_for_quota: earliest budget reset
    updated_at: datetime = Field(default_factory=utcnow)


class Segment(_Model):
    """One contiguous execution of a job on one backend session."""

    id: str = Field(default_factory=lambda: new_id("seg"))
    job_id: str
    index: int = Field(ge=0)
    backend: str
    state: SegmentState = SegmentState.LAUNCHING
    handle_id: str | None = None
    gpu_model: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    deadline_at: datetime | None = None
    exit_code: int | None = None
    start_checkpoint_seq: int | None = None
    end_checkpoint_seq: int | None = None
    handle_extra: dict[str, str] = Field(default_factory=dict)
    outcome: str | None = None  # completed | handoff | crashed | failed_user | ...
    message: str = ""
    # "main" runs the job; a "worker" only computes chunks of a parallel region (Phase 9).
    role: str = "main"
    # Set when the segment runs on paid GPU time: its price, so its cost is known, and what
    # pays it: credit the user bought on the platform, or their card (spec §21.3).
    paid_usd_per_h: float | None = Field(default=None, gt=0)
    paid_from: Literal["credit", "card"] | None = None
    region: str | None = None  # worker: id of the parallel chunks region it helps with


class CheckpointPointer(_Model):
    """Contents of ``jobs/<job_id>/checkpoints/LATEST.json`` (spec §8.1).

    A checkpoint exists only once this pointer refers to it.
    """

    seq: int = Field(ge=0)
    files: list[str]
    sha256s: dict[str, str]
    safe_point_id: str
    created_at: datetime
    segment_id: str

    @model_validator(mode="after")
    def _checksums_cover_files(self) -> CheckpointPointer:
        if set(self.files) != set(self.sha256s):
            raise ValueError("sha256s must have exactly one entry per file")
        return self


class BackendCaps(_Model):
    """Static capabilities of a backend (spec §10.1)."""

    gpu_types: list[str] = Field(default_factory=list)
    max_session_s: int = Field(gt=0)
    max_concurrent: int = Field(ge=1)
    needs_user_action: bool = False


class ReadyStatus(_Model):
    """Result of ``Backend.check_ready``."""

    ready: bool
    reason: str = ""


class LaunchHandle(_Model):
    """Opaque reference to a launched backend session."""

    backend: str
    handle_id: str
    extra: dict[str, str] = Field(default_factory=dict)


class Progress(_Model):
    cell: int | None = None
    # loop_id -> (completed iterations, total or None when unknown)
    loop: dict[str, tuple[int, int | None]] = Field(default_factory=dict)


class Rate(_Model):
    unit: str
    per_hour: float = Field(ge=0)


class ParallelStatus(_Model):
    """A ``relay.chunks(..., parallel=True)`` region the main session is in (Phase 9)."""

    region: str
    total: int = Field(ge=0)
    done: int = Field(ge=0)
    ready: bool = False  # a checkpoint inside the region exists: workers can start


class StatusFile(_Model):
    """Contents of ``jobs/<job_id>/status.json`` written by the runtime (spec §12.2).

    This, not the backend's own status, is the source of truth for job progress.
    """

    job_id: str
    segment_id: str
    state: RuntimeState
    heartbeat_at: datetime
    last_checkpoint_seq: int | None = None
    progress: Progress = Field(default_factory=Progress)
    rate: Rate | None = None
    message: str = ""
    parallel: ParallelStatus | None = None
