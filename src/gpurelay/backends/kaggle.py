"""Kaggle backend (spec §10.4) — automated only through the official ``kaggle`` package.

* One **private** kernel per job (``<user>/gpurelay-<job id>``); each segment pushes a new
  version with ``kernel-metadata.json`` + a generated ``bootstrap.py`` and Kaggle's
  ``session_timeout_seconds`` set to the segment deadline.
* The bootstrap reads the storage token from the user's private secrets dataset (attached as
  a data source), installs the pinned runtime wheel from storage, and runs the runtime.
  Secrets never appear in the kernel source or logs (hard rule 5).
* Preflight: refuses to push while the job's kernel is queued or running (no duplicate
  sessions burning quota).
* Progress comes from ``status.json`` in storage; Kaggle's status only tells running vs
  ended.

VERIFY items (docs/verify.md): session limits, weekly quota, accelerator names
(``machine_shape``), dataset mount paths, cancel API, behaviour at the time limit.
"""

from __future__ import annotations

import json
import logging
import string
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from gpurelay.backends.common import session_settings
from gpurelay.config import KaggleSettings, RuntimeSettings
from gpurelay.credentials import kaggle_credentials_present
from gpurelay.errors import BackendBusyError, GpuRelayError
from gpurelay.models import (
    BackendCaps,
    BackendRunState,
    DataSourceKind,
    Job,
    LaunchHandle,
    ReadyStatus,
    Segment,
)

log = logging.getLogger(__name__)

SECRETS_FILE = "gpurelay-secrets.json"
KERNEL_PREFIX = "gpurelay-"
# Heartbeats are commits on the Hub: keep them sparse (VERIFY: Hub commit rate limits).
BOOT_ALLOWANCE_S = 120.0  # VERIFY: Kaggle session start + bootstrap time
PUSH_SETTLE_S = 180.0  # VERIFY: delay until the status reflects a newly pushed version

_STATUS_MAP = {
    "queued": BackendRunState.QUEUED,
    "new_script": BackendRunState.QUEUED,
    "running": BackendRunState.RUNNING,
    "complete": BackendRunState.FINISHED,
    "error": BackendRunState.ERROR,
    "cancel_requested": BackendRunState.ERROR,
    "cancel_acknowledged": BackendRunState.ERROR,
}


class KaggleError(GpuRelayError):
    """A Kaggle operation failed or was refused."""


class DuplicateSessionError(KaggleError, BackendBusyError):
    """The job's kernel is already queued or running."""


class KaggleClient(Protocol):
    """The few Kaggle operations GPU Relay needs (wrapped for testability)."""

    @property
    def username(self) -> str: ...

    def push(self, folder: Path, timeout_s: int | None, accelerator: str | None) -> dict[str, Any]:
        """Push a kernel folder; returns {'version': int|None, 'url': str, 'error': str}."""
        ...

    def status(self, kernel_ref: str) -> tuple[str, str]:
        """(status name in lower snake case, failure message). ``'missing'`` if no kernel."""
        ...

    def output_log(self, kernel_ref: str, dest: Path) -> str | None:
        """The latest run's log text, if any."""
        ...

    def create_dataset(self, folder: Path) -> None:
        """Create a new **private** dataset from ``folder`` (with dataset-metadata.json)."""
        ...

    def dataset_exists(self, dataset_ref: str) -> bool: ...

    def replace_dataset(self, folder: Path) -> None:
        """New version of an existing private dataset; older versions are deleted."""
        ...


class OfficialKaggleClient:
    """:class:`KaggleClient` over ``kaggle.api.KaggleApi`` (imported lazily and safely)."""

    def __init__(self) -> None:
        if not kaggle_credentials_present():
            raise KaggleError(
                "no Kaggle credentials: create a token at https://www.kaggle.com/settings/api "
                "and save it to ~/.kaggle/access_token (or run `kaggle auth login`)"
            )
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
        except ImportError as exc:
            raise KaggleError(
                "the kaggle package is missing: pip install 'gpurelay[kaggle]'"
            ) from exc
        except SystemExit as exc:  # the client exits on invalid credentials
            raise KaggleError("Kaggle authentication failed (check your token)") from exc
        self._api = api

    @property
    def username(self) -> str:
        value = self._api.get_config_value("username")
        if not value:
            raise KaggleError("could not determine the Kaggle username")
        return str(value)

    def push(self, folder: Path, timeout_s: int | None, accelerator: str | None) -> dict[str, Any]:
        response = self._api.kernels_push(
            str(folder), str(timeout_s) if timeout_s else None, accelerator
        )
        return {
            "version": getattr(response, "versionNumber", None)
            or getattr(response, "version_number", None),
            "url": getattr(response, "url", "") or "",
            "error": getattr(response, "error", "") or "",
        }

    def status(self, kernel_ref: str) -> tuple[str, str]:
        try:
            response = self._api.kernels_status(kernel_ref)
        except Exception as exc:
            text = str(exc)
            if "404" in text or "not found" in text.lower():
                return "missing", ""
            # Kaggle answers 403 "Permission 'kernels.get' was denied" for kernels that do not
            # exist yet. Our kernels always belong to this account, so 403 means "missing".
            own = kernel_ref.lower().startswith(self.username.lower() + "/")
            if own and ("403" in text or "permission" in text.lower()):
                return "missing", ""
            raise
        status = getattr(response, "status", None)
        name = getattr(status, "name", str(status)).lower()
        return name, str(getattr(response, "failure_message", "") or "")

    def output_log(self, kernel_ref: str, dest: Path) -> str | None:
        files, _token = self._api.kernels_output(kernel_ref, str(dest), force=True, quiet=True)
        logs = [f for f in files if str(f).endswith(".log")]
        return Path(logs[0]).read_text(encoding="utf-8", errors="replace") if logs else None

    def create_dataset(self, folder: Path) -> None:
        response = self._api.dataset_create_new(str(folder), public=False, quiet=True)
        error = getattr(response, "error", "")
        if error:
            raise KaggleError(f"dataset creation failed: {error}")

    def dataset_exists(self, dataset_ref: str) -> bool:
        try:
            self._api.dataset_status(dataset_ref)
        except Exception:
            return False
        return True

    def replace_dataset(self, folder: Path) -> None:
        response = self._api.dataset_create_version(
            str(folder), "gpurelay: rotate secret", quiet=True, delete_old_versions=True
        )
        error = getattr(response, "error", "")
        if error:
            raise KaggleError(f"dataset update failed: {error}")


# ====================================================================== rendering


def kernel_slug(job_id: str) -> str:
    return f"{KERNEL_PREFIX}{job_id}".lower()


def kernel_metadata(
    *,
    username: str,
    job: Job,
    secrets_dataset: str,
    enable_gpu: bool,
    code_file: str = "bootstrap.py",
    slug: str | None = None,
) -> dict[str, Any]:
    """``kernel-metadata.json``. Always private (hard rule 5)."""
    slug = slug or kernel_slug(job.id)
    datasets = [secrets_dataset] + [d.ref for d in job.data if d.kind is DataSourceKind.KAGGLE]
    return {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "keywords": [],
    }


def render_bootstrap(
    *,
    repo_id: str,
    job_id: str,
    segment_id: str,
    wheel: str,
    deadline_s: float,
    settings: RuntimeSettings,
) -> str:
    template = string.Template(
        resources.files("gpurelay.templates")
        .joinpath("kaggle_bootstrap.py.tmpl")
        .read_text("utf-8")
    )
    return template.substitute(
        repo_id=repo_id,
        job_id=job_id,
        segment_id=segment_id,
        wheel=wheel,
        deadline_s=repr(float(deadline_s)),
        settings_literal=repr(settings.model_dump(mode="json")),
        secrets_file=SECRETS_FILE,
    )


# ====================================================================== backend


@dataclass
class KaggleBackend:
    """Backend over the official Kaggle client. ``client_factory`` is called lazily."""

    settings: KaggleSettings
    runtime: RuntimeSettings
    storage_uri: str
    work_root: Path
    wheel_for: Any  # Callable[[str], str | None]: job id -> runtime wheel file name in storage
    client_factory: Any = OfficialKaggleClient
    clock: Any = time.time
    name: str = "kaggle"
    automated: bool = True

    def __post_init__(self) -> None:
        self._client: KaggleClient | None = None

    @property
    def client(self) -> KaggleClient:
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def capabilities(self) -> BackendCaps:
        return BackendCaps(
            gpu_types=[self.settings.accelerator] if self.settings.enable_gpu else [],
            max_session_s=int(self.settings.max_session_s),
            max_concurrent=self.settings.max_concurrent,
        )

    def check_ready(self) -> ReadyStatus:
        if not self.storage_uri.startswith("hf:"):
            return ReadyStatus(ready=False, reason="Kaggle sessions need Hugging Face storage")
        if not self.settings.secrets_dataset:
            return ReadyStatus(ready=False, reason="no secrets dataset (run `gpurelay init`)")
        if not kaggle_credentials_present():
            return ReadyStatus(ready=False, reason="no Kaggle credentials")
        return ReadyStatus(ready=True)

    def _ref(self, job_id: str) -> str:
        return f"{self.client.username}/{kernel_slug(job_id)}"

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        # A parallel-chunks worker gets its own kernel, next to the job's main kernel.
        ref = self._ref(job.id) + ("-w" if segment.role == "worker" else "")
        status, _message = self.client.status(ref)
        if status in ("queued", "running", "new_script"):
            raise DuplicateSessionError(f"{ref} is already {status}; not pushing a second session")
        wheel = self.wheel_for(job.id)
        if not wheel:
            raise KaggleError(f"job {job.id} has no runtime wheel in its bundle")
        assert self.settings.secrets_dataset is not None
        timeout = int(min(deadline_s, self.settings.max_session_s))
        # Kaggle's clock starts before the bootstrap (container start, pip install): give
        # the runtime's own deadline a head start so it hands off before Kaggle kills it.
        relay_deadline = max(timeout - BOOT_ALLOWANCE_S, 0.8 * timeout)
        folder = self.work_root / job.id / segment.id
        folder.mkdir(parents=True, exist_ok=True)
        metadata = kernel_metadata(
            username=self.client.username,
            job=job,
            secrets_dataset=self.settings.secrets_dataset,
            enable_gpu=self.settings.enable_gpu,
            slug=ref.split("/", 1)[1],
        )
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
        (folder / "bootstrap.py").write_text(
            render_bootstrap(
                repo_id=self.storage_uri.removeprefix("hf:"),
                job_id=job.id,
                segment_id=segment.id,
                wheel=wheel,
                deadline_s=relay_deadline,
                settings=session_settings(self.runtime),
            ),
            "utf-8",
        )
        accelerator = self.settings.accelerator if self.settings.enable_gpu else None
        result = self.client.push(folder, timeout, accelerator)
        if result.get("error"):
            raise KaggleError(f"kernel push failed: {result['error']}")
        log.info("pushed %s version %s (%s)", ref, result.get("version"), result.get("url"))
        return LaunchHandle(
            backend=self.name,
            handle_id=ref,
            extra={
                "ref": ref,
                "version": str(result.get("version") or ""),
                "url": str(result.get("url") or ""),
                "pushed_at": repr(self.clock()),
            },
        )

    def poll(self, handle: LaunchHandle) -> BackendRunState:
        try:
            status, _message = self.client.status(handle.handle_id)
        except Exception as exc:
            log.warning("kaggle status for %s failed: %s", handle.handle_id, exc)
            return BackendRunState.UNKNOWN
        state = _STATUS_MAP.get(status, BackendRunState.UNKNOWN)
        pushed_at = float(handle.extra.get("pushed_at", "0") or 0)
        if state in (BackendRunState.FINISHED, BackendRunState.ERROR) and (
            self.clock() - pushed_at < PUSH_SETTLE_S
        ):
            # Right after a push the status may still describe the previous version.
            return BackendRunState.QUEUED
        return state

    def cancel(self, handle: LaunchHandle) -> None:
        # VERIFY: the public client has no kernel-session cancel; the controller pauses or
        # cancels jobs through control.json (handoff at the next safe point) instead.
        log.warning(
            "Kaggle sessions cannot be cancelled through the API; %s stops at its next safe "
            "point or its time limit",
            handle.handle_id,
        )

    def fetch_console_log(self, handle: LaunchHandle) -> str | None:
        dest = self.work_root / "_logs" / handle.handle_id.replace("/", "_")
        try:
            return self.client.output_log(handle.handle_id, dest)
        except Exception as exc:
            log.warning("could not fetch the Kaggle log of %s: %s", handle.handle_id, exc)
            return None
