"""Operations behind the CLI: submit, control requests, logs, fetch, quota corrections."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from gpurelay.bundle.manifest import StagedData
from gpurelay.bundle.packager import upload_bundle
from gpurelay.config import Config, gpurelay_home
from gpurelay.constants import STATE_DB_FILENAME
from gpurelay.controller.daemon import Controller
from gpurelay.controller.db import StateDB
from gpurelay.controller.registry import build_backends
from gpurelay.errors import ConfigError, GpuRelayError
from gpurelay.models import DataSpec, Job, JobKind, JobState, SegmentState
from gpurelay.pathutil import safe_join
from gpurelay.runtime.checkpoint import load_latest_checkpoint, read_latest_meta
from gpurelay.storage.base import Storage
from gpurelay.storage.layout import final_prefix, latest_key, segment_log_key
from gpurelay.storage.localfs import LocalFSStorage


class ServiceError(GpuRelayError):
    """A user-facing error from a CLI operation."""


@dataclass
class Context:
    config: Config
    home: Path
    db: StateDB
    storage: Storage
    controller: Controller


def open_storage_from_config(config: Config, home: Path) -> Storage:
    if config.storage.kind == "localfs":
        return LocalFSStorage(config.storage.localfs_root or home / "store")
    if config.storage.kind == "gdrive":
        from gpurelay.credentials import gdrive_credentials
        from gpurelay.storage.gdrive import open_gdrive

        info = gdrive_credentials()
        if not config.storage.gdrive_folder_id or info is None:
            raise ConfigError("Google Drive storage is not set up: run `gpurelay init --gdrive`")
        return open_gdrive(config.storage.gdrive_folder_id, info)
    if not config.storage.hf_repo_id:
        raise ConfigError(
            "no storage configured yet. Run one of:\n"
            "  gpurelay init --local   # this computer only, no accounts\n"
            "  gpurelay init           # Hugging Face storage (needed for Kaggle and Colab)"
        )
    from gpurelay.credentials import hf_token
    from gpurelay.storage.hf_hub import HFHubStorage

    token = hf_token()
    if not token:
        raise ConfigError("no Hugging Face token: set HF_TOKEN or run `gpurelay init`")
    return HFHubStorage(config.storage.hf_repo_id, token=token)


def open_context(config: Config, home: Path | None = None) -> Context:
    home = home or gpurelay_home()
    db = StateDB(home / STATE_DB_FILENAME)
    storage = open_storage_from_config(config, home)
    backends = build_backends(config, storage, home / "sessions")
    controller = Controller(
        config=config, db=db, storage=storage, backends=backends, outputs_root=home / "outputs"
    )
    return Context(config, home, db, storage, controller)


def submit_job(
    ctx: Context,
    path: Path,
    *,
    name: str | None = None,
    priority: int = 0,
    requirements: Path | None = None,
    data: list[DataSpec] | None = None,
    estimate_h: float | None = None,
    prefer: str | None = None,
    accept_restart_only: bool = False,
    extra_files: Mapping[str, Path] | None = None,
    max_paid_usd: float = 0.0,
    staged: list[StagedData] | None = None,
    retry_of: str | None = None,
    fix_id: str | None = None,
) -> Job:
    if prefer is not None and prefer not in ctx.controller.backends:
        raise ServiceError(
            f"unknown backend {prefer!r}; configured: {', '.join(ctx.controller.backends)}"
        )
    job = Job(
        name=name or path.stem,
        kind=JobKind.NOTEBOOK if path.suffix == ".ipynb" else JobKind.SCRIPT,
        entry=path.name,
        priority=priority,
        requirements=str(requirements) if requirements else None,
        data=list(data or []),
        estimate_h=estimate_h,
        prefer_backend=prefer,
        accept_restart_only=accept_restart_only,
        max_paid_usd=max_paid_usd,
        retry_of=retry_of,
        fix_id=fix_id,
    )
    manifest = upload_bundle(
        ctx.storage, job.id, path, extra_files, requirements=requirements, data=data,
        staged=staged,
    )  # fmt: skip
    job = job.model_copy(update={"bundle_hash": manifest.content_hash})
    ctx.db.put_job(job)
    ctx.db.add_event("submitted", f"{path.name} ({manifest.content_hash[:12]})",
                     now=time.time(), job_id=job.id)  # fmt: skip
    return job


@dataclass
class ColabSessionInfo:
    notebook: Path
    secret_name: str
    repo_id: str
    job_state: str
    segment_index: int | None
    waiting: bool
    running: bool


def colab_session(ctx: Context, ref: str) -> ColabSessionInfo:
    """Write/refresh the job's Colab resume notebook and describe what the user should do."""
    job = resolve_job(ctx, ref)
    slot = ctx.controller.backends.get("colab_manual")
    if slot is None:
        raise ServiceError(
            "the Colab backend is off: set `enabled = true` under [backends.colab_manual] in "
            f"{ctx.home / 'config.toml'}, then restart `gpurelay up`"
        )
    ready = slot.backend.check_ready()
    if not ready.ready:
        raise ServiceError(f"Colab: {ready.reason}")
    if job.state.is_terminal:
        raise ServiceError(f"{job.id} is {job.state.value}")
    notebook = slot.backend.write_notebook(job)
    live = [
        s
        for s in ctx.db.segments(job.id)
        if s.backend == "colab_manual"
        and s.state in (SegmentState.AWAITING_USER, SegmentState.RUNNING)
    ]
    seg = live[-1] if live else None
    return ColabSessionInfo(
        notebook=notebook,
        secret_name=slot.backend.settings.secret_name,
        repo_id=slot.backend.storage_uri.removeprefix("hf:"),
        job_state=job.state.value,
        segment_index=seg.index if seg else None,
        waiting=seg is not None and seg.state is SegmentState.AWAITING_USER,
        running=seg is not None and seg.state is SegmentState.RUNNING,
    )


def resolve_job(ctx: Context, ref: str) -> Job:
    job = ctx.db.find_job(ref)
    if job is None:
        raise ServiceError(f"no job matches {ref!r}")
    return job


def request(ctx: Context, ref: str, action: str) -> Job:
    """Record a pause/resume/cancel request for the controller to apply. Resume also retries a
    job that failed for platform reasons (``failed_infra``), from its last checkpoint; a job
    whose own code failed stays failed."""
    job = resolve_job(ctx, ref)
    retry = action == "resume" and job.state is JobState.FAILED_INFRA
    if job.state.is_terminal and not retry:
        raise ServiceError(f"{job.id} is {job.state.value}")
    if action == "resume" and job.state not in (JobState.PAUSED, JobState.FAILED_INFRA):
        raise ServiceError(f"{job.id} is not paused ({job.state.value})")
    job = job.model_copy(update={"requested": action})
    ctx.db.put_job(job)
    ctx.db.add_event(f"request_{action}", f"{action} requested", now=time.time(), job_id=job.id)
    return job


def merged_log(ctx: Context, job: Job) -> str:
    """Final log if completed, else the committed log so far (from the latest checkpoint)."""
    final = ctx.storage.get_bytes(f"{final_prefix(job.id)}/log.txt")
    if final is not None:
        return final.decode("utf-8", errors="replace")
    meta = read_latest_meta(ctx.storage, job.id)  # small: no state download
    if meta is None and ctx.storage.exists(latest_key(job.id)):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="gpurelay-logs-") as tmp:
            loaded = load_latest_checkpoint(ctx.storage, job.id, Path(tmp))
            meta = loaded.meta if loaded is not None else None
    if meta is None:
        return ""
    parts = []
    for entry in meta.get("runtime", {}).get("chain", []):
        data = ctx.storage.get_bytes(segment_log_key(job.id, entry["segment"])) or b""
        parts.append(data[: int(entry["log_bytes"])])
    return b"".join(parts).decode("utf-8", errors="replace")


def fetch_outputs(ctx: Context, job: Job, dest: Path) -> list[Path]:
    if job.state is not JobState.COMPLETED:
        raise ServiceError(f"{job.id} is {job.state.value}, not completed")
    prefix = final_prefix(job.id) + "/"
    written = []
    for key in ctx.storage.list(prefix):
        target = safe_join(dest, key[len(prefix) :])
        ctx.storage.get_file(key, target)
        written.append(target)
    if not written:
        raise ServiceError(f"no outputs stored for {job.id}")
    return written


def correct_quota(
    ctx: Context, backend: str, *, remaining_h: float | None, used_h: float | None
) -> None:
    if backend not in ctx.controller.backends:
        raise ServiceError(f"unknown backend {backend!r}")
    if (remaining_h is None) == (used_h is None):
        raise ServiceError("give exactly one of --remaining or --used")
    ctx.db.add_correction(
        backend, remaining_h=remaining_h, used_h=used_h, now=time.time(), note="user correction"
    )
