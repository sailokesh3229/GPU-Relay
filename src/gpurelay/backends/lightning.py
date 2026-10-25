"""Lightning AI backend (spec §10.6) — automated only through the official ``lightning_sdk``.

Gated by hard rule 4: ``[backends.lightning]`` needs ``enabled = true`` *and*
``terms_reviewed = true`` (the owner read the terms-of-service findings, docs/verify.md #10).

* One Lightning **job** per segment (``gpurelay-<job id>-<segment index>``), running a
  container image with a generated bootstrap that installs the job's pinned runtime wheel
  from storage and runs the runtime with a deadline.
* The storage token is passed in the job's environment (``HF_TOKEN``), never in the command.
* Lightning sets no session limit, so the bootstrap enforces a hard limit of its own and the
  controller's budget is the account's monthly free credits (an estimate, VERIFY #11).
* Progress comes from ``status.json`` in storage; the job status only tells running vs ended.
"""

from __future__ import annotations

import base64
import logging
import string
import time
from dataclasses import dataclass
from importlib import resources
from typing import Any, Protocol

from gpurelay.backends.common import session_settings
from gpurelay.config import LightningSettings, RuntimeSettings
from gpurelay.credentials import hf_token
from gpurelay.errors import BackendBusyError, GpuRelayError
from gpurelay.models import BackendCaps, BackendRunState, Job, LaunchHandle, ReadyStatus, Segment

log = logging.getLogger(__name__)

BOOT_ALLOWANCE_S = 300.0  # VERIFY: image pull + runtime install before the runtime starts
HARD_LIMIT_GRACE_S = 900.0  # the bootstrap kills a segment this long after its deadline

_STATUS_MAP = {
    "Pending": BackendRunState.QUEUED,
    "Running": BackendRunState.RUNNING,
    "Stopping": BackendRunState.RUNNING,
    "Completed": BackendRunState.FINISHED,
    "Stopped": BackendRunState.ERROR,
    "Failed": BackendRunState.ERROR,
    "NotCreated": BackendRunState.UNKNOWN,
}


class LightningError(GpuRelayError):
    """A Lightning AI job could not be started."""


class LightningClient(Protocol):
    """The calls GPU Relay makes, all through the official SDK."""

    def run(
        self,
        *,
        name: str,
        machine: str,
        command: str,
        image: str,
        teamspace: str | None,
        env: dict[str, str],
        interruptible: bool,
    ) -> dict[str, str]: ...

    def status(self, name: str, teamspace: str | None) -> str: ...

    def stop(self, name: str, teamspace: str | None) -> None: ...

    def logs(self, name: str, teamspace: str | None) -> str: ...


class OfficialLightningClient:
    """``lightning_sdk.Job`` (credentials: ``LIGHTNING_USER_ID`` / ``LIGHTNING_API_KEY``)."""

    def __init__(self) -> None:
        import lightning_sdk  # optional extra: gpurelay[lightning-ai]

        self._sdk = lightning_sdk

    def run(
        self,
        *,
        name: str,
        machine: str,
        command: str,
        image: str,
        teamspace: str | None,
        env: dict[str, str],
        interruptible: bool,
    ) -> dict[str, str]:
        job = self._sdk.Job.run(
            name=name,
            machine=getattr(self._sdk.Machine, machine),
            command=command,
            image=image,
            teamspace=teamspace,
            env=env,
            interruptible=interruptible,
        )
        return {"name": str(job.name), "link": str(getattr(job, "link", "") or "")}

    def _job(self, name: str, teamspace: str | None) -> Any:
        return self._sdk.Job(name, teamspace=teamspace)

    def status(self, name: str, teamspace: str | None) -> str:
        try:
            return str(self._job(name, teamspace).status)
        except RuntimeError:  # the job no longer exists
            return "NotCreated"

    def stop(self, name: str, teamspace: str | None) -> None:
        self._job(name, teamspace).stop()

    def logs(self, name: str, teamspace: str | None) -> str:
        return str(self._job(name, teamspace).logs)


def job_name(job_id: str, segment: Segment) -> str:
    role = "-w" if segment.role == "worker" else ""
    return f"gpurelay-{job_id}-{segment.index}{role}"


def render_bootstrap(
    *,
    repo_id: str,
    job_id: str,
    segment_id: str,
    wheel: str,
    deadline_s: float,
    hard_limit_s: float,
    settings: RuntimeSettings,
) -> str:
    template = string.Template(
        resources.files("gpurelay.templates")
        .joinpath("lightning_bootstrap.py.tmpl")
        .read_text("utf-8")
    )
    return template.substitute(
        job_id=job_id,
        segment_id=segment_id,
        repo_id_py=repr(repo_id),
        job_id_py=repr(job_id),
        segment_id_py=repr(segment_id),
        wheel_py=repr(wheel),
        deadline_s=repr(float(deadline_s)),
        hard_limit_s=repr(float(hard_limit_s)),
        settings_literal=repr(settings.model_dump(mode="json")),
    )


def bootstrap_command(source: str) -> str:
    """A shell command that runs ``source`` (base64: no quoting issues, no secrets in it)."""
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return 'python -c "import base64;exec(base64.b64decode(' + repr(encoded) + ').decode())"'


@dataclass
class LightningBackend:
    """Backend over the official Lightning SDK. ``client_factory`` is called lazily."""

    hub_sessions = True  # sessions run with ``session_settings`` (Hub commit budget)

    settings: LightningSettings
    runtime: RuntimeSettings
    storage_uri: str
    wheel_for: Any  # Callable[[str], str | None]
    client_factory: Any = OfficialLightningClient
    token_source: Any = hf_token
    credentials_present: Any = None  # Callable[[], bool]; default: LIGHTNING_* env vars
    clock: Any = time.time
    name: str = "lightning"
    automated: bool = True

    def __post_init__(self) -> None:
        self._client: LightningClient | None = None

    @property
    def client(self) -> LightningClient:
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def capabilities(self) -> BackendCaps:
        return BackendCaps(
            gpu_types=[self.settings.machine],
            max_session_s=int(self.settings.max_session_s),
            max_concurrent=self.settings.max_concurrent,
        )

    def check_ready(self) -> ReadyStatus:
        import importlib.util
        import os

        if not self.storage_uri.startswith("hf:"):
            return ReadyStatus(ready=False, reason="Lightning jobs need Hugging Face storage")
        if importlib.util.find_spec("lightning_sdk") is None:
            return ReadyStatus(ready=False, reason="pip install 'gpurelay[lightning-ai]'")
        present = (
            self.credentials_present()
            if self.credentials_present is not None
            else bool(os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID"))
        )
        if not present:
            return ReadyStatus(
                ready=False,
                reason="set LIGHTNING_USER_ID and LIGHTNING_API_KEY (lightning.ai settings)",
            )
        return ReadyStatus(ready=True)

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        name = job_name(job.id, segment)
        if self.client.status(name, self.settings.teamspace) in ("Pending", "Running"):
            raise BackendBusyError(f"Lightning job {name} is already running")
        wheel = self.wheel_for(job.id)
        if not wheel:
            raise LightningError(f"job {job.id} has no runtime wheel in its bundle")
        token = self.token_source()
        if not token:
            raise LightningError("no Hugging Face token (HF_TOKEN or `gpurelay init`)")
        limit = min(deadline_s, self.settings.max_session_s)
        source = render_bootstrap(
            repo_id=self.storage_uri.removeprefix("hf:"),
            job_id=job.id,
            segment_id=segment.id,
            wheel=wheel,
            deadline_s=max(limit - BOOT_ALLOWANCE_S, 0.8 * limit),
            hard_limit_s=limit + HARD_LIMIT_GRACE_S,
            settings=session_settings(self.runtime),
        )
        result = self.client.run(
            name=name,
            machine=self.settings.machine,
            command=bootstrap_command(source),
            image=self.settings.image,
            teamspace=self.settings.teamspace,
            env={"HF_TOKEN": token},
            interruptible=self.settings.interruptible,
        )
        actual = result.get("name") or name
        log.info("started Lightning job %s (%s)", actual, result.get("link", ""))
        return LaunchHandle(
            backend=self.name,
            handle_id=actual,
            extra={"link": result.get("link", ""), "started_at": repr(self.clock())},
        )

    def poll(self, handle: LaunchHandle) -> BackendRunState:
        try:
            status = self.client.status(handle.handle_id, self.settings.teamspace)
        except Exception as exc:
            log.warning("lightning status for %s failed: %s", handle.handle_id, exc)
            return BackendRunState.UNKNOWN
        return _STATUS_MAP.get(status, BackendRunState.UNKNOWN)

    def cancel(self, handle: LaunchHandle) -> None:
        try:
            self.client.stop(handle.handle_id, self.settings.teamspace)
        except Exception as exc:
            log.warning("could not stop Lightning job %s: %s", handle.handle_id, exc)

    def fetch_console_log(self, handle: LaunchHandle) -> str | None:
        try:
            return self.client.logs(handle.handle_id, self.settings.teamspace)
        except Exception as exc:
            log.warning("could not fetch the log of %s: %s", handle.handle_id, exc)
            return None
