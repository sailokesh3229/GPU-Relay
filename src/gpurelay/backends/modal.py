"""Modal backend (plan P12): GPU sessions through Modal's official Python SDK only.

* One Modal **Sandbox** per segment (named ``gpurelay-<job id>-<segment index>``) in the
  ``gpurelay`` app, on the configured GPU, running a container image with a generated
  bootstrap (shared with Lightning AI) that installs the job's pinned runtime wheel from
  storage and runs the runtime with a deadline.
* The storage token reaches the sandbox as a Modal secret (``HF_TOKEN``), never in the
  command or any file.
* Budget: Modal grants dollars, not hours (Starter plan: $30 of compute a month; a T4 costs
  about $0.59 an hour), so the budget is in dollars (plan P11). Both figures are settings:
  check modal.com/pricing and correct them in config.toml.
* Gated like Lightning AI (hard rule 4): ``terms_reviewed = true`` is required.
* Sandboxes live at most 24 hours; segments hand off long before (``max_session_s``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from gpurelay.backends.common import session_settings
from gpurelay.backends.lightning import bootstrap_command, job_name, render_bootstrap
from gpurelay.config import ModalSettings, RuntimeSettings
from gpurelay.credentials import hf_token
from gpurelay.errors import BackendBusyError, GpuRelayError
from gpurelay.models import BackendCaps, BackendRunState, Job, LaunchHandle, ReadyStatus, Segment

log = logging.getLogger(__name__)

APP_NAME = "gpurelay"
BOOT_ALLOWANCE_S = 300.0  # image pull + runtime install before the runtime starts
HARD_LIMIT_GRACE_S = 600.0  # the sandbox is killed this long after the segment's deadline
SANDBOX_MAX_S = 24 * 3600  # Modal's limit for one sandbox


class ModalError(GpuRelayError):
    """A Modal sandbox could not be started."""


class ModalClient(Protocol):
    """The calls GPU Relay makes, all through the official SDK."""

    def run(
        self, *, name: str, gpu: str, command: str, image: str, timeout_s: int,
        env: dict[str, str],
    ) -> dict[str, str]: ...  # fmt: skip

    def status(self, sandbox_id: str) -> str: ...  # "running" | "finished:<code>" | "missing"

    def stop(self, sandbox_id: str) -> None: ...

    def logs(self, sandbox_id: str) -> str: ...


class OfficialModalClient:
    """``modal.Sandbox`` (credentials: ``modal token new``, i.e. ~/.modal.toml, or the
    ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` environment variables)."""

    def __init__(self) -> None:
        import modal  # optional extra: gpurelay[modal]

        self._modal = modal
        self._app: Any = None

    def _get_app(self) -> Any:
        if self._app is None:
            self._app = self._modal.App.lookup(APP_NAME, create_if_missing=True)
        return self._app

    def run(
        self, *, name: str, gpu: str, command: str, image: str, timeout_s: int,
        env: dict[str, str],
    ) -> dict[str, str]:  # fmt: skip
        modal = self._modal
        sandbox = modal.Sandbox.create(
            "bash", "-lc", command,
            app=self._get_app(),
            name=name,
            image=modal.Image.from_registry(image),
            secrets=[modal.Secret.from_dict(dict(env))],  # sent to Modal, not in the command
            gpu=gpu,
            timeout=int(timeout_s),
        )  # fmt: skip
        return {"id": str(sandbox.object_id)}

    def status(self, sandbox_id: str) -> str:
        try:
            sandbox = self._modal.Sandbox.from_id(sandbox_id)
        except Exception:
            return "missing"
        code = sandbox.poll()
        return "running" if code is None else f"finished:{code}"

    def stop(self, sandbox_id: str) -> None:
        self._modal.Sandbox.from_id(sandbox_id).terminate()

    def logs(self, sandbox_id: str) -> str:
        sandbox = self._modal.Sandbox.from_id(sandbox_id)
        if sandbox.poll() is None:
            return ""  # streams can only be read whole once the sandbox has finished
        return str(sandbox.stdout.read()) + str(sandbox.stderr.read())


def modal_credentials_present() -> bool:
    import os
    from pathlib import Path

    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    config = Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml")
    try:
        return "token_secret" in config.read_text(encoding="utf-8")
    except OSError:
        return False


@dataclass
class ModalBackend:
    """Backend over the official Modal SDK. ``client_factory`` is called lazily."""

    hub_sessions = True  # sessions run with ``session_settings`` (Hub commit budget)

    settings: ModalSettings
    runtime: RuntimeSettings
    storage_uri: str
    wheel_for: Any  # Callable[[str], str | None]
    client_factory: Any = OfficialModalClient
    token_source: Any = hf_token
    credentials_present: Any = modal_credentials_present
    clock: Any = time.time
    name: str = "modal"
    automated: bool = True

    def __post_init__(self) -> None:
        self._client: ModalClient | None = None

    @property
    def client(self) -> ModalClient:
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def capabilities(self) -> BackendCaps:
        return BackendCaps(
            gpu_types=[self.settings.gpu],
            max_session_s=int(self.settings.max_session_s),
            max_concurrent=self.settings.max_concurrent,
        )

    def check_ready(self) -> ReadyStatus:
        import importlib.util

        if not self.storage_uri.startswith("hf:"):
            return ReadyStatus(ready=False, reason="Modal sessions need Hugging Face storage")
        if importlib.util.find_spec("modal") is None:
            return ReadyStatus(ready=False, reason="pip install 'gpurelay[modal]'")
        if not self.credentials_present():
            return ReadyStatus(
                ready=False, reason="no Modal token: run `modal token new` (or set MODAL_TOKEN_*)"
            )
        return ReadyStatus(ready=True)

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        name = job_name(job.id, segment)
        previous = getattr(segment, "handle_id", None)
        if previous and self.client.status(previous) == "running":
            raise BackendBusyError(f"Modal sandbox {previous} is already running")
        wheel = self.wheel_for(job.id)
        if not wheel:
            raise ModalError(f"job {job.id} has no runtime wheel in its bundle")
        token = self.token_source()
        if not token:
            raise ModalError("no Hugging Face token (HF_TOKEN or `gpurelay init`)")
        limit = min(deadline_s, self.settings.max_session_s)
        source = render_bootstrap(
            repo_id=self.storage_uri.removeprefix("hf:"),
            job_id=job.id,
            segment_id=segment.id,
            wheel=wheel,
            deadline_s=max(limit - BOOT_ALLOWANCE_S, 0.8 * limit),
            hard_limit_s=limit,
            settings=session_settings(self.runtime),
            backend="modal",
        )
        result = self.client.run(
            name=name,
            gpu=self.settings.gpu,
            command=bootstrap_command(source),
            image=self.settings.image,
            timeout_s=int(min(limit + HARD_LIMIT_GRACE_S, SANDBOX_MAX_S)),
            env={"HF_TOKEN": token},
        )
        sandbox_id = result["id"]
        log.info("started Modal sandbox %s (%s, %s)", sandbox_id, name, self.settings.gpu)
        return LaunchHandle(
            backend=self.name,
            handle_id=sandbox_id,
            extra={"name": name, "started_at": repr(self.clock())},
        )

    def poll(self, handle: LaunchHandle) -> BackendRunState:
        try:
            status = self.client.status(handle.handle_id)
        except Exception as exc:
            log.warning("modal status for %s failed: %s", handle.handle_id, exc)
            return BackendRunState.UNKNOWN
        if status == "running":
            return BackendRunState.RUNNING
        if status.startswith("finished:"):
            return BackendRunState.FINISHED if status == "finished:0" else BackendRunState.ERROR
        return BackendRunState.UNKNOWN

    def cancel(self, handle: LaunchHandle) -> None:
        try:
            self.client.stop(handle.handle_id)
        except Exception as exc:
            log.warning("could not stop Modal sandbox %s: %s", handle.handle_id, exc)

    def fetch_console_log(self, handle: LaunchHandle) -> str | None:
        try:
            return self.client.logs(handle.handle_id) or None
        except Exception as exc:
            log.warning("could not fetch the log of %s: %s", handle.handle_id, exc)
            return None


__all__ = ["ModalBackend", "ModalClient", "OfficialModalClient", "modal_credentials_present"]
