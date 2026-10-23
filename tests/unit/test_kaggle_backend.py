"""Kaggle backend with a fake client: metadata, secrets hygiene, status mapping, duplicate
guard, init/setup, and the requirements policy (no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpurelay.backends.kaggle import (
    SECRETS_FILE,
    DuplicateSessionError,
    KaggleBackend,
    kernel_metadata,
    kernel_slug,
)
from gpurelay.config import Config, KaggleSettings, RuntimeSettings
from gpurelay.controller.setup import setup_kaggle, setup_storage, validate_session_token
from gpurelay.errors import BackendBusyError, ConfigError, HardRuleViolation
from gpurelay.models import BackendRunState, DataSpec, Job, JobKind, Segment
from gpurelay.runtime.requirements import plan_requirements

SECRET = "hf_SUPERSECRET_token_value"


@dataclass
class FakeClient:
    username: str = "lokesh"
    statuses: dict[str, str] = field(default_factory=dict)
    pushed: list[tuple[Path, int | None, str | None]] = field(default_factory=list)
    datasets: dict[str, dict[str, str]] = field(default_factory=dict)

    def push(self, folder: Path, timeout_s: int | None, accelerator: str | None) -> dict[str, Any]:
        self.pushed.append((folder, timeout_s, accelerator))
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        self.statuses[meta["id"]] = "queued"
        return {
            "version": len(self.pushed),
            "url": f"https://kaggle.com/code/{meta['id']}",
            "error": "",
        }

    def status(self, kernel_ref: str) -> tuple[str, str]:
        return self.statuses.get(kernel_ref, "missing"), ""

    def output_log(self, kernel_ref: str, dest: Path) -> str | None:
        return "log text"

    def create_dataset(self, folder: Path) -> None:
        meta = json.loads((folder / "dataset-metadata.json").read_text())
        self.datasets[meta["id"]] = {p.name: p.read_text() for p in folder.iterdir()}

    def dataset_exists(self, dataset_ref: str) -> bool:
        return dataset_ref in self.datasets

    def replace_dataset(self, folder: Path) -> None:
        self.create_dataset(folder)


def _backend(tmp_path: Path, client: FakeClient, **settings: Any) -> KaggleBackend:
    kaggle = KaggleSettings(
        enabled=True, username="lokesh", secrets_dataset="lokesh/gpurelay-secrets",
        max_session_s=3600, **settings,
    )  # fmt: skip
    return KaggleBackend(
        settings=kaggle,
        runtime=RuntimeSettings(),
        storage_uri="hf:lokesh/gpurelay-store",
        work_root=tmp_path / "kaggle",
        wheel_for=lambda _job: "gpurelay-0.1.0.dev0-py3-none-any.whl",
        client_factory=lambda: client,
    )


def _job() -> Job:
    return Job(
        name="vae", kind=JobKind.SCRIPT, entry="vae.py",
        data=[DataSpec.parse("kaggle:owner/mnist"), DataSpec.parse("url:https://x.org/a.zip")],
    )  # fmt: skip


def test_kernel_metadata_is_private_and_attaches_datasets() -> None:
    job = _job()
    meta = kernel_metadata(username="u", job=job, secrets_dataset="u/gpurelay-secrets",
                           enable_gpu=True)  # fmt: skip
    assert meta["is_private"] is True
    assert meta["enable_internet"] is True and meta["enable_gpu"] is True
    assert meta["dataset_sources"] == ["u/gpurelay-secrets", "owner/mnist"]
    assert meta["id"] == f"u/{meta['title']}" == f"u/{kernel_slug(job.id)}"


def test_launch_writes_no_secrets_and_valid_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", SECRET)
    client = FakeClient()
    backend = _backend(tmp_path, client)
    job = _job()
    seg = Segment(job_id=job.id, index=0, backend="kaggle")
    handle = backend.launch(job, seg, deadline_s=1800)
    folder, timeout, accelerator = client.pushed[0]
    assert timeout == 1800 and accelerator == "NvidiaTeslaT4"
    assert handle.handle_id == f"lokesh/{kernel_slug(job.id)}"
    for path in folder.iterdir():
        assert SECRET not in path.read_text()
    source = (folder / "bootstrap.py").read_text()
    compile(source, "bootstrap.py", "exec")
    assert SECRETS_FILE in source and "hf:" in source and job.id in source
    assert "repr(START + DEADLINE_S)" in source
    assert 'token.startswith("hf_")' in source  # a misplaced Kaggle token fails clearly
    deadline = float(source.split("DEADLINE_S = ")[1].splitlines()[0])
    assert deadline < 1800  # hands off before Kaggle's own timeout


def test_cpu_only_sessions(tmp_path: Path) -> None:
    client = FakeClient()
    backend = _backend(tmp_path, client, enable_gpu=False)
    job = _job()
    backend.launch(job, Segment(job_id=job.id, index=0, backend="kaggle"), 600)
    folder, _timeout, accelerator = client.pushed[0]
    assert accelerator is None
    assert json.loads((folder / "kernel-metadata.json").read_text())["enable_gpu"] is False


def test_duplicate_guard(tmp_path: Path) -> None:
    client = FakeClient()
    backend = _backend(tmp_path, client)
    job = _job()
    backend.launch(job, Segment(job_id=job.id, index=0, backend="kaggle"), 600)
    with pytest.raises(DuplicateSessionError) as info:
        backend.launch(job, Segment(job_id=job.id, index=1, backend="kaggle"), 600)
    assert isinstance(info.value, BackendBusyError)
    assert len(client.pushed) == 1


@pytest.mark.parametrize(
    ("status", "state"),
    [
        ("queued", BackendRunState.QUEUED),
        ("running", BackendRunState.RUNNING),
        ("complete", BackendRunState.FINISHED),
        ("error", BackendRunState.ERROR),
        ("cancel_acknowledged", BackendRunState.ERROR),
        ("weird", BackendRunState.UNKNOWN),
    ],
)
def test_status_mapping(tmp_path: Path, status: str, state: BackendRunState) -> None:
    client = FakeClient(statuses={"lokesh/k": status})
    backend = _backend(tmp_path, client)
    from gpurelay.models import LaunchHandle

    assert backend.poll(LaunchHandle(backend="kaggle", handle_id="lokesh/k")) is state


def test_status_right_after_push_is_not_trusted(tmp_path: Path) -> None:
    client = FakeClient()
    backend = _backend(tmp_path, client)
    now = [1000.0]
    backend.clock = lambda: now[0]
    job = _job()
    handle = backend.launch(job, Segment(job_id=job.id, index=0, backend="kaggle"), 600)
    client.statuses[handle.handle_id] = "complete"  # still the previous version's status
    assert backend.poll(handle) is BackendRunState.QUEUED
    now[0] += 200
    assert backend.poll(handle) is BackendRunState.FINISHED


def test_check_ready_requires_hub_storage_and_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    backend = _backend(tmp_path, FakeClient())
    assert backend.check_ready().ready
    backend.storage_uri = "localfs:/tmp"
    assert "Hugging Face" in backend.check_ready().reason
    backend.storage_uri = "hf:a/b"
    backend.settings.secrets_dataset = None
    assert "secrets" in backend.check_ready().reason


# ---------------------------------------------------------------- init / setup


class FakeHfApi:
    def __init__(self, private: bool = True) -> None:
        self.repos: dict[str, bool] = {}
        self.private = private

    def whoami(self, token: str | None = None) -> dict[str, str]:
        if token is not None and token != SECRET:
            raise PermissionError("401 Unauthorized")
        return {"name": "lokesh"}

    def repo_info(self, repo_id: str, repo_type: str, token: str | None = None) -> Any:
        if repo_id not in self.repos:
            raise KeyError(repo_id)
        return type("Info", (), {"private": self.repos[repo_id]})()

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool) -> None:
        self.repos[repo_id] = private and self.private


def test_setup_storage_creates_private_repo() -> None:
    config = Config()
    api = FakeHfApi()
    result = setup_storage(config, api)
    assert result.repo_id == "lokesh/gpurelay-store" and result.created
    assert config.storage.kind == "hf_hub" and config.storage.hf_repo_id == "lokesh/gpurelay-store"
    assert not setup_storage(config, api).created


def test_setup_storage_refuses_public_repo() -> None:
    api = FakeHfApi()
    api.repos["lokesh/gpurelay-store"] = False
    with pytest.raises(HardRuleViolation, match="public"):
        setup_storage(Config(), api)


def test_setup_kaggle_creates_private_secrets_dataset() -> None:
    config = Config()
    client = FakeClient()
    result = setup_kaggle(config, client, session_token=SECRET, repo_id="lokesh/gpurelay-store")
    assert result.secrets_dataset == "lokesh/gpurelay-secrets" and result.created
    stored = client.datasets["lokesh/gpurelay-secrets"]
    assert json.loads(stored[SECRETS_FILE])["hf_token"] == SECRET
    assert config.backends.kaggle.enabled and config.backends.kaggle.username == "lokesh"
    assert not setup_kaggle(config, client, session_token=SECRET, repo_id="r").created


def test_setup_kaggle_rotate_replaces_the_secret() -> None:
    config = Config()
    client = FakeClient()
    setup_kaggle(config, client, session_token="KGAT_wrong", repo_id="r")
    result = setup_kaggle(config, client, session_token=SECRET, repo_id="r", rotate=True)
    assert result.created
    stored = client.datasets["lokesh/gpurelay-secrets"]
    assert json.loads(stored[SECRETS_FILE])["hf_token"] == SECRET


def test_session_token_must_be_a_working_hf_token() -> None:
    api = FakeHfApi()
    api.repos["lokesh/gpurelay-store"] = True
    assert validate_session_token(SECRET, "lokesh/gpurelay-store", api) == "lokesh"
    with pytest.raises(ConfigError, match="Kaggle token"):
        validate_session_token("KGAT_0123456789abcdef", "lokesh/gpurelay-store", api)
    with pytest.raises(ConfigError, match="cannot access"):
        validate_session_token("hf_other", "lokesh/gpurelay-store", api)
    with pytest.raises(ConfigError, match="cannot access"):
        validate_session_token(SECRET, "lokesh/missing", api)


def test_setup_kaggle_enforces_one_account() -> None:
    config = Config()
    config.backends.kaggle.username = "someone-else"
    with pytest.raises(HardRuleViolation, match="one account"):
        setup_kaggle(config, FakeClient(), session_token=SECRET, repo_id="r")


# ---------------------------------------------------------------- requirements policy


def test_requirements_keep_preinstalled_frameworks(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    versions = {"torch": "2.4.0", "tensorflow": "2.16.1"}

    def fake_version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    plan = plan_requirements(
        ["torch>=2.0", "torchvision", "tensorflow==2.16.0", "torch==2.3.1  # old",
         "numpy==1.26.4", "--extra-index-url https://x", "", "# comment"]
    )  # fmt: skip
    assert plan.kept == ["torch>=2.0", "tensorflow==2.16.0"]
    assert plan.install == [
        "torchvision",  # not preinstalled here
        "torch==2.3.1",  # different major.minor pinned
        "numpy==1.26.4",
        "--extra-index-url https://x",
    ]


def test_official_client_treats_403_on_own_kernel_as_missing() -> None:
    """Kaggle returns 403 'Permission kernels.get was denied' for kernels that do not exist."""
    from gpurelay.backends.kaggle import OfficialKaggleClient

    class Api:
        def get_config_value(self, name: str) -> str:
            return "lokesh"

        def kernels_status(self, ref: str) -> Any:
            raise ValueError(f"Cannot access kernel '{ref}' (Permission 'kernels.get' was denied).")

    client = OfficialKaggleClient.__new__(OfficialKaggleClient)
    client._api = Api()
    assert client.status("lokesh/gpurelay-job-1") == ("missing", "")
    with pytest.raises(ValueError):
        client.status("someone-else/kernel")  # not ours: a real permission problem
