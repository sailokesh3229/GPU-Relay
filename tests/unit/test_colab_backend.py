"""Colab (manual) backend: resume notebook generation, the notebook's own guard logic run
against fake Colab/Hub modules, session requests, and the hard-rule-2 compliance check."""

from __future__ import annotations

import ast
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import nbformat
import pytest

import gpurelay.backends.colab_manual as colab_module
from gpurelay.backends.colab_manual import ColabManualBackend, render_notebook
from gpurelay.config import (
    BackendsSettings,
    ColabSettings,
    Config,
    LocalSettings,
    RuntimeSettings,
    StorageSettings,
)
from gpurelay.controller.registry import build_backends
from gpurelay.models import BackendRunState, Job, JobKind, Segment
from gpurelay.storage.layout import latest_key, manual_session_key, status_key
from gpurelay.storage.localfs import LocalFSStorage

SECRET = "hf_SUPERSECRET_token_value"
REPO = "lokesh/gpurelay-store"
WHEEL = "gpurelay-0.1.0.dev0-py3-none-any.whl"


def _job() -> Job:
    return Job(name="vae", kind=JobKind.SCRIPT, entry="vae.py")


def _backend(tmp_path: Path, **settings: Any) -> ColabManualBackend:
    return ColabManualBackend(
        settings=ColabSettings(enabled=True, **settings),
        runtime=RuntimeSettings(),
        storage=LocalFSStorage(tmp_path / "store"),
        storage_uri=f"hf:{REPO}",
        work_root=tmp_path / "colab",
        wheel_for=lambda _job_id: WHEEL,
        clock=lambda: 1000.0,
    )


def _notebook(job: Job) -> Any:
    return render_notebook(
        job=job, repo_id=REPO, wheel=WHEEL, secret_name="GPURELAY_HF_TOKEN",
        settings=RuntimeSettings(),
    )  # fmt: skip


def _code(notebook: Any) -> list[str]:
    return [c.source for c in notebook.cells if c.cell_type == "code"]


# ---------------------------------------------------------------- notebook


def test_notebook_is_valid_generic_and_has_no_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", SECRET)
    backend = _backend(tmp_path)
    job = _job()
    path = backend.write_notebook(job)
    assert path.name == f"resume_{job.id}.ipynb"
    notebook = nbformat.read(str(path), as_version=4)
    nbformat.validate(notebook)
    text = path.read_text("utf-8")
    assert SECRET not in text
    assert notebook.cells[0].cell_type == "markdown"
    assert "GPURELAY_HF_TOKEN" in notebook.cells[0].source
    code = "\n".join(_code(notebook))
    for source in _code(notebook):
        compile(source, "cell", "exec")
    assert job.id in code and REPO in code and WHEEL in code
    assert "userdata.get(SECRET_NAME)" in code
    # One notebook per job: the segment comes from storage, not from the notebook.
    assert "seg-" not in code
    assert 'PYTHONHASHSEED="0"' in code


def test_notebook_quotes_values_safely() -> None:
    job = Job(name='my "job" $HOME', kind=JobKind.SCRIPT, entry="a.py")
    notebook = render_notebook(job=job, repo_id=REPO, wheel=WHEEL, secret_name='we"ird',
                               settings=RuntimeSettings())  # fmt: skip
    namespace: dict[str, Any] = {}
    first = _code(notebook)[0]
    exec(first.split("class GpuRelayStop")[0], namespace)
    assert namespace["SECRET_NAME"] == 'we"ird' and namespace["REPO_ID"] == REPO


# ---------------------------------------------------------------- the notebook's guard


class _NotFound(Exception):
    def __init__(self) -> None:
        super().__init__("404")
        self.response = types.SimpleNamespace(status_code=404)


def _run_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: LocalFSStorage, secret: Any
) -> dict[str, Any]:
    """Execute the notebook's first two code cells with fake Colab and Hub modules."""

    def userdata_get(name: str) -> str:
        if isinstance(secret, Exception):
            raise secret
        return str(secret)

    def hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        assert repo_id == REPO and kwargs["token"] == secret
        data = storage.get_bytes(filename)
        if data is None:
            raise _NotFound()
        out = tmp_path / "downloads" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=userdata_get)  # type: ignore[attr-defined]
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    cells = _code(_notebook(_job_for(storage)))
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for source in cells[:2]:
        exec(compile(source, "cell", "exec"), namespace)
    return namespace


_JOB = _job()


def _job_for(_storage: LocalFSStorage) -> Job:
    return _JOB


def _put(storage: LocalFSStorage, key: str, value: dict[str, Any]) -> None:
    storage.put_bytes(json.dumps(value).encode(), key)


def _request(storage: LocalFSStorage, segment: str = "seg-a", seq: int | None = 3) -> None:
    _put(storage, manual_session_key(_JOB.id),
         {"segment_id": segment, "deadline_s": 7200.0, "start_checkpoint_seq": seq})  # fmt: skip
    if seq is not None:
        _put(storage, latest_key(_JOB.id), {"seq": seq})


def _status(storage: LocalFSStorage, segment: str, state: str, age_s: float) -> None:
    beat = datetime.now(UTC) - timedelta(seconds=age_s)
    _put(storage, status_key(_JOB.id),
         {"segment_id": segment, "state": state, "heartbeat_at": beat.isoformat()})  # fmt: skip


def _stopped(exc: BaseException, match: str) -> None:
    assert type(exc).__name__ == "GpuRelayStop", exc
    assert match in str(exc), exc


def test_guard_starts_the_requested_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalFSStorage(tmp_path / "store")
    _request(storage)
    _status(storage, "seg-previous", "handoff", age_s=30)  # e.g. the last Kaggle segment
    ns = _run_guard(tmp_path, monkeypatch, storage, SECRET)
    assert ns["SEGMENT_ID"] == "seg-a" and ns["SEQ"] == 3 and ns["DEADLINE_S"] == 7200.0
    import os

    assert os.environ["HF_TOKEN"] == SECRET


def test_guard_first_segment_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalFSStorage(tmp_path / "store")
    _request(storage, seq=None)
    assert _run_guard(tmp_path, monkeypatch, storage, SECRET)["SEQ"] is None


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda s: None, "not asking for a Colab session"),
        (lambda s: (_request(s), _put(s, latest_key(_JOB.id), {"seq": 5})), "moved on"),
        (lambda s: (_request(s), _status(s, "seg-a", "running", 5)), "already running"),
        (lambda s: (_request(s), _status(s, "seg-a", "handoff", 5)), "already ran (handoff)"),
        (lambda s: (_request(s), _status(s, "seg-a", "running", 9999)), "already ran"),
    ],
    ids=["no-request", "ran-elsewhere", "running-in-another-tab", "rerun", "stale"],
)
def test_guard_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: Any, message: str
) -> None:
    storage = LocalFSStorage(tmp_path / "store")
    setup(storage)
    with pytest.raises(Exception) as info:
        _run_guard(tmp_path, monkeypatch, storage, SECRET)
    _stopped(info.value, message)


@pytest.mark.parametrize(
    ("secret", "message"),
    [
        (KeyError("GPURELAY_HF_TOKEN"), "add a Colab secret named GPURELAY_HF_TOKEN"),
        ("KGAT_0123456789", "must hold a Hugging Face token"),
    ],
)
def test_guard_checks_the_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret: Any, message: str
) -> None:
    storage = LocalFSStorage(tmp_path / "store")
    _request(storage)
    with pytest.raises(Exception) as info:
        _run_guard(tmp_path, monkeypatch, storage, secret)
    _stopped(info.value, message)


# ---------------------------------------------------------------- backend


def test_launch_requests_a_session_and_cancel_withdraws_it(tmp_path: Path) -> None:
    backend = _backend(tmp_path, max_session_s=3600)
    job = _job()
    seg = Segment(job_id=job.id, index=2, backend="colab_manual", start_checkpoint_seq=7)
    handle = backend.launch(job, seg, deadline_s=99999)
    assert Path(handle.extra["notebook"]).exists()
    request = json.loads(backend.storage.get_bytes(manual_session_key(job.id)) or b"")
    assert request["segment_id"] == seg.id and request["start_checkpoint_seq"] == 7
    assert request["deadline_s"] == 3600.0  # capped by the Colab session limit
    assert backend.poll(handle) is BackendRunState.UNKNOWN
    assert backend.fetch_console_log(handle) is None

    other = Segment(job_id=job.id, index=3, backend="colab_manual")
    backend.cancel(handle.model_copy(update={"extra": {**handle.extra, "segment_id": other.id}}))
    assert backend.storage.exists(manual_session_key(job.id))  # not ours: kept
    backend.cancel(handle)
    assert not backend.storage.exists(manual_session_key(job.id))


def test_capabilities_and_readiness(tmp_path: Path) -> None:
    backend = _backend(tmp_path, gpu_label="T4")
    caps = backend.capabilities()
    assert caps.needs_user_action and caps.max_concurrent == 1 and caps.gpu_types == ["T4"]
    assert backend.check_ready().ready
    backend.storage_uri = "localfs:/x"
    assert "Hugging Face" in backend.check_ready().reason


def test_registry_adds_a_manual_slot(tmp_path: Path) -> None:
    config = Config(
        storage=StorageSettings(kind="localfs", localfs_root=tmp_path / "store"),
        backends=BackendsSettings(local=LocalSettings(enabled=False),
                                  colab_manual=ColabSettings(enabled=True)),
    )  # fmt: skip
    slots = build_backends(config, LocalFSStorage(tmp_path / "store"), tmp_path / "sessions")
    slot = slots["colab_manual"]
    assert slot.automated is False and slot.backend.automated is False
    assert slot.budget.hours is None and slot.max_concurrent == 1


# ---------------------------------------------------------------- hard rule 2


FORBIDDEN_MODULES = {
    "subprocess", "webbrowser", "socket", "selenium", "playwright", "pyppeteer", "paramiko",
    "asyncssh", "fabric", "pexpect", "pyngrok", "requests", "httpx", "urllib", "aiohttp",
    "websocket", "websockets", "multiprocessing", "threading",
}  # fmt: skip
FORBIDDEN_WORDS = ("ssh", "tunnel", "ngrok", "cloudflared", "selenium", "playwright",
                   "webdriver", "remote_exec", "http_over_ws", "localtunnel")  # fmt: skip


def test_colab_backend_has_no_automation_paths() -> None:
    """Spec §3 rule 2: no browser automation, SSH, tunnels, or remote execution."""
    source = Path(colab_module.__file__).read_text("utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & FORBIDDEN_MODULES, imported & FORBIDDEN_MODULES
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    ).lower()
    docstring_free = code.replace("never automated", "")
    assert not [w for w in FORBIDDEN_WORDS if w in docstring_free]
    assert ColabManualBackend.automated is False


def test_resume_notebook_has_no_remote_access_code() -> None:
    code = "\n".join(_code(_notebook(_job()))).lower()
    assert not [w for w in FORBIDDEN_WORDS if w in code]
    assert "colab.research.google.com" not in code  # the notebook never opens Colab itself


# ---------------------------------------------------------------- `gpurelay colab`


def test_colab_session_explains_what_to_do(tmp_path: Path) -> None:
    from gpurelay.controller.service import ServiceError, colab_session, open_context
    from gpurelay.models import SegmentState

    def config(enabled: bool) -> Config:
        return Config(
            storage=StorageSettings(kind="localfs", localfs_root=tmp_path / "store"),
            backends=BackendsSettings(local=LocalSettings(enabled=False),
                                      colab_manual=ColabSettings(enabled=enabled)),
        )  # fmt: skip

    ctx = open_context(config(False), tmp_path / "home")
    job = ctx.db.put_job(_job())
    with pytest.raises(ServiceError, match="Colab backend is off"):
        colab_session(ctx, job.id)

    ctx = open_context(config(True), tmp_path / "home")
    with pytest.raises(ServiceError, match="Hugging Face storage"):
        colab_session(ctx, job.id)

    backend = ctx.controller.backends["colab_manual"].backend
    backend.storage_uri = f"hf:{REPO}"
    backend.wheel_for = lambda _job_id: WHEEL
    info = colab_session(ctx, job.id)
    assert info.notebook.exists() and not info.waiting and not info.running
    assert info.secret_name == "GPURELAY_HF_TOKEN" and info.repo_id == REPO

    seg = Segment(job_id=job.id, index=0, backend="colab_manual", state=SegmentState.AWAITING_USER)
    ctx.db.put_segment(seg)
    info = colab_session(ctx, job.id)
    assert info.waiting and info.segment_index == 0
