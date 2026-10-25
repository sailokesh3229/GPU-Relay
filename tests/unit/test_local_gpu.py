"""Your own GPU (plan P15) and local jobs from the packaged app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpurelay.backends import local_env
from gpurelay.config import Config, StorageSettings
from gpurelay.controller import registry
from gpurelay.controller.gpus import speed_of
from gpurelay.storage.localfs import LocalFSStorage


def _run(stdout: str, code: int = 0) -> Any:
    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[1:] == ["--query-gpu=name", "--format=csv,noheader"]
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="")

    return run


def test_detect_gpu_reads_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "C:/nvidia-smi.exe")
    gpu = local_env.detect_gpu(run=_run("NVIDIA GeForce RTX 3060 Laptop GPU\n"))
    assert gpu == "NVIDIA GeForce RTX 3060 Laptop GPU" and speed_of(gpu) == 1.2
    assert local_env.detect_gpu(run=_run("", code=9)) is None  # driver error
    monkeypatch.setattr(local_env.shutil, "which", lambda name: None)
    assert local_env.detect_gpu(run=_run("ignored")) is None  # no NVIDIA driver at all


def test_registry_counts_a_detected_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = Config(storage=StorageSettings(kind="localfs", localfs_root=tmp_path / "store"))
    cfg.backends.kaggle.enabled = False
    monkeypatch.setattr(registry, "detect_local_gpu", lambda: "NVIDIA GeForce RTX 4090")
    slots = registry.build_backends(cfg, LocalFSStorage(tmp_path / "store"), tmp_path / "s")
    assert slots["local"].gpu_label == "NVIDIA GeForce RTX 4090" and slots["local"].speed == 4.0
    cfg.backends.local.detect_gpu = False
    slots = registry.build_backends(cfg, LocalFSStorage(tmp_path / "store"), tmp_path / "s")
    assert slots["local"].gpu_label is None and slots["local"].speed == 0.05


def test_a_normal_install_runs_jobs_with_this_python(tmp_path: Path) -> None:
    assert local_env.runtime_python(tmp_path) == sys.executable


def test_the_packaged_app_explains_a_missing_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(local_env, "frozen", lambda: True)
    monkeypatch.setattr(local_env, "system_python", lambda: None)
    with pytest.raises(local_env.NoPythonError, match=r"python.org"):
        local_env.runtime_python(tmp_path)


def test_the_packaged_app_builds_its_job_environment_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import gpurelay

    monkeypatch.setattr(local_env, "frozen", lambda: True)
    monkeypatch.setattr(local_env, "system_python", lambda: ["py", "-3"])
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> None:
        calls.append(cmd)
        if cmd[-2:] == ["venv", str(tmp_path / "local-runtime")]:
            python = local_env._venv_python(tmp_path / "local-runtime")
            python.parent.mkdir(parents=True)
            python.write_text("")

    monkeypatch.setattr(local_env.subprocess, "run", fake_run)
    python = local_env.runtime_python(tmp_path)
    assert python.endswith(("python.exe", "python"))
    assert calls[0][:3] == ["py", "-3", "-m"] and "pip" in calls[1]
    assert calls[1][-1].endswith(".whl")
    assert (tmp_path / "local-runtime" / local_env.MARKER).read_text() == gpurelay.__version__
    assert local_env.runtime_python(tmp_path) == python and len(calls) == 2  # cached


def test_local_backend_resolves_its_interpreter_on_first_launch(tmp_path: Path) -> None:
    from gpurelay.backends.local import LocalBackend

    resolved: list[int] = []
    backend = LocalBackend(LocalFSStorage(tmp_path / "s"), tmp_path / "r",
                           python_resolver=lambda: resolved.append(1) or sys.executable,
                           ready_check=lambda: "install Python")  # fmt: skip
    status = backend.check_ready()
    assert not status.ready and status.reason == "install Python"
    assert resolved == []  # nothing is installed until a job really runs here
