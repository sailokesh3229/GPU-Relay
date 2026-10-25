"""Modal backend (plan P12) with a fake client: no secrets in the command, deadlines within
Modal's limits, status mapping, readiness, the terms gate and the dollar budget (no network)."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpurelay.backends.modal import SANDBOX_MAX_S, ModalBackend, modal_credentials_present
from gpurelay.config import Config, ModalSettings, RuntimeSettings, StorageSettings
from gpurelay.controller.registry import build_backends
from gpurelay.errors import BackendBusyError, GpuRelayError
from gpurelay.models import BackendRunState, Job, JobKind, LaunchHandle, Segment
from gpurelay.storage import LocalFSStorage

SECRET = "hf_SUPERSECRET_modal_token"


@dataclass
class FakeModal:
    statuses: dict[str, str] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)

    def run(self, **kwargs: Any) -> dict[str, str]:
        self.runs.append(kwargs)
        sandbox_id = f"sb-{len(self.runs)}"
        self.statuses[sandbox_id] = "running"
        return {"id": sandbox_id}

    def status(self, sandbox_id: str) -> str:
        return self.statuses.get(sandbox_id, "missing")

    def stop(self, sandbox_id: str) -> None:
        self.stopped.append(sandbox_id)

    def logs(self, sandbox_id: str) -> str:
        return "sandbox log"


def _backend(client: FakeModal, **kw: Any) -> ModalBackend:
    return ModalBackend(
        settings=ModalSettings(enabled=True, terms_reviewed=True, **kw),
        runtime=RuntimeSettings(),
        storage_uri="hf:me/gpurelay-store",
        wheel_for=lambda _job: "gpurelay-1.0.0-py3-none-any.whl",
        client_factory=lambda: client,
        token_source=lambda: SECRET,
        credentials_present=lambda: True,
    )


def _job() -> Job:
    return Job(name="vae", kind=JobKind.SCRIPT, entry="vae.py")


def _segment(job: Job, index: int = 0, handle: str | None = None) -> Segment:
    return Segment(job_id=job.id, index=index, backend="modal", handle_id=handle)


def _decoded(command: str) -> str:
    encoded = re.search(r"b64decode\('([^']+)'\)", command)
    assert encoded is not None
    return base64.b64decode(encoded.group(1)).decode()


def test_launch_runs_a_sandbox_without_secrets_in_the_command() -> None:
    client = FakeModal()
    job = _job()
    handle = _backend(client, gpu="L4").launch(job, _segment(job), deadline_s=3 * 3600)
    [run] = client.runs
    assert handle.handle_id == "sb-1" and handle.extra["name"] == f"gpurelay-{job.id}-0"
    assert run["gpu"] == "L4" and run["env"] == {"HF_TOKEN": SECRET}
    assert SECRET not in run["command"]
    source = _decoded(run["command"])
    assert SECRET not in source and "\"--backend\", 'modal'" in source
    assert "gpurelay-1.0.0-py3-none-any.whl" in source
    assert run["timeout_s"] <= 3 * 3600 + 600  # dies soon after the deadline, never long after


def test_session_length_respects_settings_and_modals_24_hours() -> None:
    client = FakeModal()
    job = _job()
    _backend(client, max_session_s=24 * 3600).launch(job, _segment(job), deadline_s=99 * 3600)
    assert client.runs[0]["timeout_s"] == SANDBOX_MAX_S
    with pytest.raises(ValueError):
        ModalSettings(max_session_s=25 * 3600)


def test_status_mapping_cancel_and_logs() -> None:
    client = FakeModal()
    backend = _backend(client)
    handle = LaunchHandle(backend="modal", handle_id="sb-9")
    for status, state in [("running", BackendRunState.RUNNING),
                          ("finished:0", BackendRunState.FINISHED),
                          ("finished:1", BackendRunState.ERROR),
                          ("missing", BackendRunState.UNKNOWN)]:  # fmt: skip
        client.statuses["sb-9"] = status
        assert backend.poll(handle) is state
    backend.cancel(handle)
    assert client.stopped == ["sb-9"]
    assert backend.fetch_console_log(handle) == "sandbox log"


def test_a_running_sandbox_is_not_started_twice() -> None:
    client = FakeModal()
    job = _job()
    client.statuses["sb-old"] = "running"
    with pytest.raises(BackendBusyError):
        _backend(client).launch(job, _segment(job, handle="sb-old"), deadline_s=3600)


def test_launch_needs_a_wheel_and_a_token() -> None:
    job = _job()
    backend = _backend(FakeModal())
    backend.wheel_for = lambda _job: None
    with pytest.raises(GpuRelayError, match="runtime wheel"):
        backend.launch(job, _segment(job), deadline_s=3600)
    backend = _backend(FakeModal())
    backend.token_source = lambda: None
    with pytest.raises(GpuRelayError, match="Hugging Face token"):
        backend.launch(job, _segment(job), deadline_s=3600)


def test_readiness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = _backend(FakeModal())
    assert backend.check_ready().ready
    backend.credentials_present = lambda: False
    assert "modal token new" in (backend.check_ready().reason or "")
    backend.storage_uri = "localfs:/x"
    assert "Hugging Face storage" in (backend.check_ready().reason or "")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / ".modal.toml"))
    assert not modal_credentials_present()
    (tmp_path / ".modal.toml").write_text('[default]\ntoken_id = "ak-x"\ntoken_secret = "as-y"\n')
    assert modal_credentials_present()


def test_modal_is_gated_by_its_terms_and_budgets_in_dollars(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="terms_reviewed"):
        ModalSettings(enabled=True)
    cfg = Config(storage=StorageSettings(kind="hf_hub", hf_repo_id="me/gpurelay-store"))
    cfg.backends.local.enabled = False
    cfg.backends.kaggle.enabled = False
    cfg.backends.modal = ModalSettings(enabled=True, terms_reviewed=True, gpu="T4",
                                       monthly_credit_usd=30.0, usd_per_hour=0.59)  # fmt: skip
    slots = build_backends(cfg, LocalFSStorage(tmp_path / "s"), tmp_path / "sessions")
    slot = slots["modal"]
    assert slot.budget.unit == "usd" and slot.budget.hours == pytest.approx(50.85, abs=0.01)
    assert slot.budget.model.value == "monthly" and slot.speed == 1.0 and slot.automated
