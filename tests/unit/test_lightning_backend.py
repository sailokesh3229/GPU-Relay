"""Lightning AI backend with a fake client (Phase 9 item 4): no secrets in the command,
status mapping, duplicate guard, readiness, registry gate (no network)."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpurelay.backends.lightning import LightningBackend, job_name
from gpurelay.config import (
    BackendsSettings,
    Config,
    LightningSettings,
    LocalSettings,
    RuntimeSettings,
    StorageSettings,
)
from gpurelay.controller.registry import build_backends
from gpurelay.errors import BackendBusyError
from gpurelay.models import BackendRunState, Job, JobKind, LaunchHandle, Segment
from gpurelay.storage import LocalFSStorage

SECRET = "hf_SUPERSECRET_token_value"


@dataclass
class FakeLightning:
    statuses: dict[str, str] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)

    def run(self, **kwargs: Any) -> dict[str, str]:
        self.runs.append(kwargs)
        self.statuses[kwargs["name"]] = "Pending"
        return {"name": kwargs["name"], "link": f"https://lightning.ai/jobs/{kwargs['name']}"}

    def status(self, name: str, teamspace: str | None) -> str:
        return self.statuses.get(name, "NotCreated")

    def stop(self, name: str, teamspace: str | None) -> None:
        self.stopped.append(name)

    def logs(self, name: str, teamspace: str | None) -> str:
        return "log text"


def _settings(**kw: Any) -> LightningSettings:
    return LightningSettings(enabled=True, terms_reviewed=True, **kw)


def _backend(client: FakeLightning, **kw: Any) -> LightningBackend:
    return LightningBackend(
        settings=_settings(**kw),
        runtime=RuntimeSettings(),
        storage_uri="hf:lokesh/gpurelay-store",
        wheel_for=lambda _job: "gpurelay-0.1.0.dev0-py3-none-any.whl",
        client_factory=lambda: client,
        token_source=lambda: SECRET,
        credentials_present=lambda: True,
    )


def _job() -> Job:
    return Job(name="vae", kind=JobKind.SCRIPT, entry="vae.py")


def test_launch_passes_the_token_only_in_the_environment() -> None:
    client = FakeLightning()
    backend = _backend(client, machine="L4", teamspace="lokesh/default")
    job = _job()
    seg = Segment(job_id=job.id, index=3, backend="lightning")
    handle = backend.launch(job, seg, deadline_s=3600)
    [run] = client.runs
    assert run["name"] == handle.handle_id == job_name(job.id, seg) == f"gpurelay-{job.id}-3"
    assert run["machine"] == "L4" and run["teamspace"] == "lokesh/default"
    assert run["env"] == {"HF_TOKEN": SECRET}
    assert SECRET not in run["command"]
    encoded = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", run["command"])
    assert encoded is not None
    source = base64.b64decode(encoded.group(1)).decode()
    compile(source, "bootstrap.py", "exec")
    assert SECRET not in source and job.id in source and seg.id in source
    deadline = float(source.split("DEADLINE_S = ")[1].splitlines()[0])
    hard = float(source.split("HARD_LIMIT_S = ")[1].splitlines()[0])
    assert deadline < 3600 < hard  # hands off before the limit; killed if stuck


def test_session_length_is_capped_by_max_session() -> None:
    client = FakeLightning()
    backend = _backend(client, max_session_s=1800)
    job = _job()
    backend.launch(job, Segment(job_id=job.id, index=0, backend="lightning"), deadline_s=99999)
    source = base64.b64decode(
        re.search(r"b64decode\('([^']+)'\)", client.runs[0]["command"]).group(1)  # type: ignore[union-attr]
    ).decode()
    assert float(source.split("DEADLINE_S = ")[1].splitlines()[0]) < 1800


@pytest.mark.parametrize(
    ("status", "state"),
    [("Pending", BackendRunState.QUEUED), ("Running", BackendRunState.RUNNING),
     ("Completed", BackendRunState.FINISHED), ("Failed", BackendRunState.ERROR),
     ("Stopped", BackendRunState.ERROR), ("NotCreated", BackendRunState.UNKNOWN)],
)  # fmt: skip
def test_status_mapping(status: str, state: BackendRunState) -> None:
    client = FakeLightning(statuses={"j": status})
    assert _backend(client).poll(LaunchHandle(backend="lightning", handle_id="j")) is state


def test_duplicate_guard_and_cancel() -> None:
    client = FakeLightning()
    backend = _backend(client)
    job = _job()
    seg = Segment(job_id=job.id, index=0, backend="lightning")
    handle = backend.launch(job, seg, 600)
    with pytest.raises(BackendBusyError):
        backend.launch(job, seg, 600)
    backend.cancel(handle)
    assert client.stopped == [handle.handle_id]
    assert backend.fetch_console_log(handle) == "log text"


def test_readiness_reasons() -> None:
    backend = _backend(FakeLightning())
    backend.storage_uri = "localfs:/x"
    assert "Hugging Face" in backend.check_ready().reason
    backend.storage_uri = "hf:x/y"
    backend.credentials_present = lambda: False
    reason = backend.check_ready().reason
    assert "LIGHTNING_API_KEY" in reason or "lightning-ai" in reason


def test_registry_adds_lightning_only_when_enabled(tmp_path: Path) -> None:
    def config(lightning: LightningSettings) -> Config:
        return Config(
            storage=StorageSettings(kind="localfs", localfs_root=tmp_path / "store"),
            backends=BackendsSettings(local=LocalSettings(enabled=False), lightning=lightning),
        )

    storage = LocalFSStorage(tmp_path / "store")
    assert "lightning" not in build_backends(config(LightningSettings()), storage, tmp_path)
    slot = build_backends(config(_settings()), storage, tmp_path)["lightning"]
    assert slot.automated and slot.budget.hours == 22.0 and slot.gpu_label == "T4"
