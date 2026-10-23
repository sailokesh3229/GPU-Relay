"""Stored relative paths never escape their directory (hardening campaign, security)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpurelay.bundle.packager import download_bundle, upload_bundle
from gpurelay.pathutil import UnsafePathError, safe_join
from gpurelay.storage import LocalFSStorage


@pytest.mark.parametrize(
    "rel", ["../x", "a/../../x", "/etc/passwd", "C:/Windows/x", "c:x", "", "a/b/..", r"..\x"]
)
def test_escapes_are_refused(tmp_path: Path, rel: str) -> None:
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, rel)


@pytest.mark.parametrize(
    "rel", ["a.txt", "dir/sub/file.bin", "..hidden", "a..b/c", r"a\b", "a//b", "./a", "a/./b"]
)
def test_normal_paths_are_kept(tmp_path: Path, rel: str) -> None:
    target = safe_join(tmp_path, rel)
    assert target.resolve().is_relative_to(tmp_path.resolve())


def test_tampered_bundle_cannot_write_outside(tmp_path: Path) -> None:
    store = LocalFSStorage(tmp_path / "store")
    script = tmp_path / "job.py"
    script.write_text("print(1)\n")
    upload_bundle(store, "job-1", script, include_wheel=False)
    key = "jobs/job-1/bundle/manifest.json"
    manifest = json.loads(store.get_bytes(key) or b"")
    sha = next(iter(manifest["files"].values()))
    manifest["files"] = {"../../escaped.py": sha}  # a tampered store
    store.put_bytes(json.dumps(manifest).encode(), key)
    with pytest.raises(Exception, match=r"unsafe|escaped|not in"):
        download_bundle(store, "job-1", tmp_path / "work" / "deep")
    assert not (tmp_path / "escaped.py").exists()
