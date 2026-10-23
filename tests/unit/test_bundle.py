"""Bundles: archive round trip, content hash, data staging, runtime wheel (Phase 5)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from gpurelay.bundle.packager import (
    BundleError,
    download_bundle,
    read_bundle_file,
    stage_data,
    upload_bundle,
)
from gpurelay.bundle.wheel import build_runtime_wheel
from gpurelay.models import DataSpec, JobKind
from gpurelay.storage import LocalFSStorage


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "data" / "nested").mkdir(parents=True)
    (root / "train.py").write_text("print('hi')\n")
    (root / "helpers.py").write_text("X = 1\n")
    (root / "requirements.txt").write_text("numpy\n")
    (root / "data" / "a.csv").write_text("1,2\n")
    (root / "data" / "nested" / "b.csv").write_text("3,4\n")
    return root


@pytest.fixture
def store(tmp_path: Path) -> LocalFSStorage:
    return LocalFSStorage(tmp_path / "store")


def test_round_trip_and_content_hash(project: Path, store: LocalFSStorage, tmp_path: Path) -> None:
    kwargs = dict(
        extra_files={"helpers.py": project / "helpers.py"},
        requirements=project / "requirements.txt",
        include_wheel=False,
    )
    first = upload_bundle(store, "job-1", project / "train.py", **kwargs)  # type: ignore[arg-type]
    second = upload_bundle(store, "job-2", project / "train.py", **kwargs)  # type: ignore[arg-type]
    assert first.kind is JobKind.SCRIPT and first.requirements == "requirements.txt"
    assert first.content_hash == second.content_hash == first.compute_hash()
    assert sorted(first.files) == ["helpers.py", "requirements.txt", "train.py"]

    work = tmp_path / "work"
    manifest = download_bundle(store, "job-1", work)
    assert manifest == first
    assert (work / "helpers.py").read_text() == "X = 1\n"
    assert read_bundle_file(store, "job-1", "train.py") == (project / "train.py").read_bytes()

    (project / "train.py").write_text("print('changed')\n")
    third = upload_bundle(store, "job-3", project / "train.py", include_wheel=False)
    assert third.content_hash != first.content_hash


def test_local_data_is_uploaded_once_and_staged(
    project: Path, store: LocalFSStorage, tmp_path: Path
) -> None:
    spec = DataSpec.parse("local:data")
    m1 = upload_bundle(store, "job-1", project / "train.py", data=[spec], include_wheel=False)
    m2 = upload_bundle(store, "job-2", project / "train.py", data=[spec], include_wheel=False)
    assert m1.data[0].cache_key == m2.data[0].cache_key
    assert len(store.list("data-cache/")) == 2
    work = tmp_path / "work"
    download_bundle(store, "job-1", work)
    assert stage_data(store, m1, work) == ["data"]
    assert (work / "data" / "nested" / "b.csv").read_text() == "3,4\n"


def test_unsupported_and_missing_data(project: Path, store: LocalFSStorage) -> None:
    manifest = upload_bundle(
        store,
        "j",
        project / "train.py",
        data=[DataSpec.parse("hf:org/ds"), DataSpec.parse("kaggle:owner/mnist-small")],
        include_wheel=False,
    )
    assert [d.target for d in manifest.data] == ["ds", "mnist-small"]
    with pytest.raises(BundleError, match="does not exist"):
        upload_bundle(store, "j", project / "train.py", data=[DataSpec.parse("local:nope")])


def test_runtime_wheel_is_deterministic_and_installable(tmp_path: Path) -> None:
    first = build_runtime_wheel(tmp_path / "w1")
    second = build_runtime_wheel(tmp_path / "w2")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as zf:
        names = zf.namelist()
    assert "gpurelay/__init__.py" in names and "gpurelay/runtime/entry.py" in names
    assert not any("__pycache__" in n for n in names)
    target = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target",
         str(target), str(first)],
        check=True,
    )  # fmt: skip
    probe = subprocess.run(
        [sys.executable, "-c", "import gpurelay, sys; print(gpurelay.__file__)"],
        env={
            "PYTHONPATH": str(target),
            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(target) in probe.stdout
