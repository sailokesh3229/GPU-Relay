"""Paid GPU platforms (spec §21, plan G3-G5): one machine per segment, deleted afterwards.

Every platform runs the same session: a short shell script that runs the generated bootstrap
(shared with Lightning AI and Modal: install the job's runtime wheel from storage, run the
segment with a deadline) under a hard ``timeout``, then ends the machine where the platform
allows it. A :class:`CloudProvider` does only what differs: start a machine, look at it,
delete it, list GPU Relay's machines, and (where the platform has them) its price and the
account's balance.

Nothing may keep billing (spec §21.3):

* the session stops itself at its hard limit (``timeout``), and ends its machine where the
  platform lets a machine do that (``self_end``);
* the controller deletes the machine as soon as the segment is over (``deletes_machines``):
  as soon as ``status.json`` says so, even if the machine is still up (Lambda keeps a machine
  "active" after its work ends, DigitalOcean and Azure bill a powered-off machine);
* a machine that never finished starting is given up after ``BOOT_LIMIT_S``;
* :meth:`CloudBackend.sweep` deletes GPU Relay's machines that no live segment owns.

Machine names are ``gpurelay-<owner>-<job id>-<segment index>``: ``owner`` comes from the
storage repo, so two GPU Relays sharing a cloud account never touch each other's machines.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from gpurelay.backends.common import session_settings
from gpurelay.backends.lightning import render_bootstrap
from gpurelay.config import CloudSettings, RuntimeSettings
from gpurelay.credentials import hf_token
from gpurelay.errors import BackendBusyError, GpuRelayError
from gpurelay.models import BackendCaps, BackendRunState, Job, LaunchHandle, ReadyStatus, Segment

log = logging.getLogger(__name__)

BOOT_ALLOWANCE_S = 900.0  # machine start + image pull + runtime install, before the deadline
BOOT_LIMIT_S = 1800.0  # a machine still starting after this long is given up
HARD_LIMIT_GRACE_S = 600.0
PRICE_REFRESH_S = 3600.0
STARTING, RUNNING, STOPPED, GONE, FAILED = "starting", "running", "stopped", "gone", "failed"


class CloudError(GpuRelayError):
    """A paid platform refused or failed a request (``status``: its HTTP status)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Machine:
    id: str
    name: str
    state: str  # STARTING | RUNNING | STOPPED (no longer working, may still exist) | GONE | FAILED
    detail: str = ""


class CloudProvider(Protocol):
    """What differs between platforms. ``env`` holds only ``HF_TOKEN``."""

    label: str  # "RunPod"
    gpu_label: str  # for the GPU speed table and the GPU-continuity tie-break
    vm: bool  # a whole machine (cloud-init / startup script) rather than a container
    self_end: str  # shell lines that end this machine from inside ("" when it cannot)

    def missing(self) -> str | None: ...  # why it cannot be used yet (no credentials), or None
    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]: ...
    def machine(self, machine_id: str) -> Machine: ...
    def delete(self, machine_id: str) -> None: ...
    def ours(self, prefix: str) -> list[Machine]: ...  # machines whose name starts with prefix
    def logs(self, machine_id: str) -> str | None: ...
    def price_usd_per_h(self) -> float | None: ...
    def balance_usd(self) -> float | None: ...


def owner_tag(storage_uri: str) -> str:
    return hashlib.sha256(storage_uri.encode("utf-8")).hexdigest()[:8]


def machine_name(owner: str, job_id: str, segment: Segment) -> str:
    role = "-w" if segment.role == "worker" else ""
    return f"gpurelay-{owner}-{job_id}-{segment.index}{role}".lower()


def session_script(
    *, name: str, bootstrap: str, hard_limit_s: float, vm: bool, self_end: str,
    env: dict[str, str] | None = None,
) -> str:  # fmt: skip
    """The shell script a machine runs. ``env``: exported here (whole machines get no other
    way to receive it: it travels in the platform's user data / startup script, which only
    the account can read); containers get it from the platform instead."""
    packed = base64.b64encode(zlib.compress(bootstrap.encode("utf-8"), 9)).decode("ascii")
    lines = ["#!/bin/bash", f"# GPU Relay session {name}: one segment of a job, then this "
             "machine ends.", "set -u", "cd /tmp"]  # fmt: skip
    for key, value in (env or {}).items():
        lines.append(f"export {key}='{value}'")
    lines.append("PY=python3")
    if vm:  # system Python refuses pip (PEP 668): a venv that still sees the image's packages
        lines += [
            "if ! python3 -m venv --system-site-packages /opt/gpurelay >/dev/null 2>&1; then",
            "  (apt-get update -q && apt-get install -y -q python3-venv python3-pip)"
            " >/dev/null 2>&1",
            "  python3 -m venv --system-site-packages /opt/gpurelay",
            "fi",
            "PY=/opt/gpurelay/bin/python",
        ]  # fmt: skip
    else:
        lines.append("command -v python3 >/dev/null || PY=python")
    lines += [
        f'timeout -k 120 {int(hard_limit_s)} "$PY" -c '
        f"\"import base64,zlib;exec(zlib.decompress(base64.b64decode('{packed}')).decode())\"",
        "code=$?",
        'echo "[gpurelay] session finished with exit code $code"',
    ]
    if self_end:
        lines.append(self_end)
    return "\n".join(lines) + "\n"


@dataclass
class CloudBackend:
    """A paid platform as a GPU Relay backend (see the module docstring)."""

    hub_sessions = True  # sessions run with ``session_settings`` (Hub commit budget)
    deletes_machines = True  # the controller deletes the machine when the segment is over

    name: str
    settings: CloudSettings
    runtime: RuntimeSettings
    storage_uri: str
    wheel_for: Callable[[str], str | None]
    provider: CloudProvider
    token_source: Callable[[], str | None] = hf_token
    clock: Callable[[], float] = time.time
    automated: bool = True
    _price: tuple[float, float] | None = field(default=None, init=False)  # (USD/h, when)

    @property
    def owner(self) -> str:
        return owner_tag(self.storage_uri)

    def capabilities(self) -> BackendCaps:
        return BackendCaps(
            gpu_types=[self.provider.gpu_label],
            max_session_s=int(self.settings.max_session_s),
            max_concurrent=self.settings.max_concurrent,
        )

    def check_ready(self) -> ReadyStatus:
        if not self.storage_uri.startswith("hf:"):
            reason = f"{self.provider.label} sessions need Hugging Face storage"
            return ReadyStatus(ready=False, reason=reason)
        missing = self.provider.missing()
        if missing:
            return ReadyStatus(ready=False, reason=missing)
        return ReadyStatus(ready=True)

    # ---------------------------------------------------------------- prices and balance

    def price_usd_per_h(self) -> float | None:
        """The configured price, else the platform's (refreshed hourly)."""
        if self.settings.usd_per_hour is not None:
            return self.settings.usd_per_hour
        now = self.clock()
        if self._price is None or now - self._price[1] > PRICE_REFRESH_S:
            try:
                value = self.provider.price_usd_per_h()
            except Exception as exc:
                log.warning("could not read the %s price: %s", self.provider.label, exc)
                value = None
            if value is not None:
                self._price = (float(value), now)
        return self._price[0] if self._price else None

    def balance_usd(self) -> float | None:
        return self.provider.balance_usd()

    # ---------------------------------------------------------------- sessions

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        name = machine_name(self.owner, job.id, segment)
        previous = getattr(segment, "handle_id", None)
        if previous and self.provider.machine(previous).state in (STARTING, RUNNING):
            raise BackendBusyError(f"{self.provider.label} machine {previous} is still running")
        wheel = self.wheel_for(job.id)
        if not wheel:
            raise CloudError(f"job {job.id} has no runtime wheel in its bundle")
        token = self.token_source()
        if not token:
            raise CloudError("no Hugging Face token (HF_TOKEN or `gpurelay init`)")
        limit = min(deadline_s, float(self.settings.max_session_s))
        bootstrap = render_bootstrap(
            repo_id=self.storage_uri.removeprefix("hf:"), job_id=job.id, segment_id=segment.id,
            wheel=wheel, deadline_s=max(limit - BOOT_ALLOWANCE_S, 0.75 * limit),
            hard_limit_s=limit, settings=session_settings(self.runtime), backend=self.name,
        )  # fmt: skip
        env = {"HF_TOKEN": token}
        script = session_script(
            name=name, bootstrap=bootstrap, hard_limit_s=limit + HARD_LIMIT_GRACE_S,
            vm=self.provider.vm, self_end=self.provider.self_end,
            env=env if self.provider.vm else None,
        )  # fmt: skip
        started = self.provider.start(name, script, env)
        log.info("started %s machine %s (%s)", self.provider.label, started["id"], name)
        extra = {"name": name, "started_at": repr(self.clock()),
                 **{k: v for k, v in started.items() if k != "id"}}  # fmt: skip
        return LaunchHandle(backend=self.name, handle_id=started["id"], extra=extra)

    def poll(self, handle: LaunchHandle) -> BackendRunState:
        try:
            machine = self.provider.machine(handle.handle_id)
        except Exception as exc:
            log.warning("%s status of %s failed: %s", self.provider.label, handle.handle_id, exc)
            return BackendRunState.UNKNOWN
        if machine.state == STARTING:
            started = float(handle.extra.get("started_at", self.clock()))
            if self.clock() - started > BOOT_LIMIT_S:
                log.warning("%s machine %s never finished starting (%s); giving up",
                            self.provider.label, handle.handle_id, machine.detail)  # fmt: skip
                return BackendRunState.ERROR
            return BackendRunState.QUEUED
        if machine.state == RUNNING:
            return BackendRunState.RUNNING
        if machine.state in (STOPPED, GONE):
            return BackendRunState.FINISHED  # status.json says how the segment ended
        return BackendRunState.ERROR

    def cancel(self, handle: LaunchHandle) -> None:
        """Delete the machine (also when the segment ended: nothing may keep billing)."""
        try:
            self.provider.delete(handle.handle_id)
        except Exception as exc:
            log.warning("could not delete %s machine %s: %s", self.provider.label,
                        handle.handle_id, exc)  # fmt: skip

    def fetch_console_log(self, handle: LaunchHandle) -> str | None:
        try:
            return self.provider.logs(handle.handle_id)
        except Exception as exc:
            log.warning("could not fetch the log of %s: %s", handle.handle_id, exc)
            return None

    def sweep(self, keep: set[str]) -> list[str]:
        """Delete this GPU Relay's machines that no live segment owns (``keep``: machine ids
        of live segments). Returns the deleted ids."""
        deleted = []
        for machine in self.provider.ours(f"gpurelay-{self.owner}-"):
            if machine.id in keep or machine.state == GONE:
                continue
            log.warning("deleting %s machine %s (%s): no live segment owns it",
                        self.provider.label, machine.id, machine.name)  # fmt: skip
            try:
                self.provider.delete(machine.id)
            except Exception as exc:
                log.warning("could not delete %s: %s", machine.id, exc)
                continue
            deleted.append(machine.id)
        return deleted


# ---------------------------------------------------------------- HTTP for REST platforms

Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@dataclass
class Rest:
    """JSON over HTTPS with a bearer token; errors never quote the token."""

    base: str
    token: str
    label: str
    transport: Transport = urllib_transport
    timeout: float = 60.0

    def call(self, method: str, path: str, body: Any = None, *, ok: tuple[int, ...] = ()) -> Any:
        import json

        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json",
                   "User-Agent": "gpurelay"}  # fmt: skip
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        url = path if path.startswith("https://") else self.base.rstrip("/") + path
        status, raw = self.transport(method, url, headers, data, self.timeout)
        if not (200 <= status < 300 or status in ok):
            text = raw.decode("utf-8", "replace")[:300].replace(self.token, "***")
            raise CloudError(f"{self.label} answered {status} to {method} {path}: {text}", status)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return {"text": raw.decode("utf-8", "replace")}
