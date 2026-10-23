"""Google Drive storage (Phase 9 item 5) over an in-memory fake Drive (no network)."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Any

import pytest

from gpurelay.config import Config
from gpurelay.controller.setup import setup_gdrive
from gpurelay.errors import StorageNotFoundError
from gpurelay.retry import RetryPolicy
from gpurelay.runtime.checkpoint import CheckpointManager, load_latest_checkpoint
from gpurelay.storage.base import Storage
from gpurelay.storage.factory import open_storage, storage_uri
from gpurelay.storage.gdrive import FOLDER_MIME, GoogleDriveStorage

FAST = RetryPolicy(attempts=4, base_delay_s=0, max_delay_s=0)


class HttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.resp = type("Resp", (), {"status": status})()


class FakeDrive:
    """Drive-like store: ids, parents, duplicate names allowed, a modification clock."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.ids = itertools.count(1)
        self.clock = itertools.count(1)
        self.fail_next: list[int] = []  # HTTP statuses to raise on the next calls
        self.calls = 0

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.fail_next:
            raise HttpError(self.fail_next.pop(0))

    def _new(self, parent: str, name: str, mime: str, data: bytes | None) -> str:
        file_id = f"id{next(self.ids)}"
        self.entries[file_id] = {
            "id": file_id, "name": name, "mimeType": mime, "parent": parent, "data": data,
            "modifiedTime": f"{next(self.clock):08d}",
        }  # fmt: skip
        return file_id

    def children(self, parent_id: str, name: str | None = None) -> list[dict[str, Any]]:
        self._maybe_fail()
        out = []
        for e in self.entries.values():
            if e["parent"] == parent_id and (name is None or e["name"] == name):
                view = {k: v for k, v in e.items() if k not in ("parent", "data")}
                if e["data"] is not None:
                    view["sha256Checksum"] = hashlib.sha256(e["data"]).hexdigest()
                out.append(view)
        return out

    def create_folder(self, parent_id: str, name: str) -> str:
        self._maybe_fail()
        return self._new(parent_id, name, FOLDER_MIME, None)

    def upload(self, parent_id: str, name: str, data: bytes, file_id: str | None) -> str:
        self._maybe_fail()
        if file_id is None:
            return self._new(parent_id, name, "application/octet-stream", data)
        entry = self.entries[file_id]
        entry["data"] = data
        entry["modifiedTime"] = f"{next(self.clock):08d}"
        return file_id

    def download(self, file_id: str) -> bytes:
        self._maybe_fail()
        if file_id not in self.entries:
            raise HttpError(404)
        data = self.entries[file_id]["data"]
        assert data is not None
        return bytes(data)

    def delete(self, file_id: str) -> None:
        self._maybe_fail()
        if self.entries.pop(file_id, None) is None:
            raise HttpError(404)


@pytest.fixture
def drive() -> FakeDrive:
    fake = FakeDrive()
    fake.entries["root-folder"] = {
        "id": "root-folder", "name": "gpurelay-store", "mimeType": FOLDER_MIME,
        "parent": "root", "data": None, "modifiedTime": "0",
    }  # fmt: skip
    return fake


@pytest.fixture
def store(drive: FakeDrive) -> GoogleDriveStorage:
    return GoogleDriveStorage("root-folder", drive, retry=FAST)


def test_protocol_and_basic_operations(store: GoogleDriveStorage, tmp_path: Path) -> None:
    assert isinstance(store, Storage)
    assert store.get_bytes("jobs/j1/status.json") is None and not store.exists("jobs/x")
    store.put_bytes(b"one", "jobs/j1/status.json")
    assert store.get_bytes("jobs/j1/status.json") == b"one" and store.exists("jobs/j1/status.json")
    src = tmp_path / "f.bin"
    src.write_bytes(b"file data")
    store.put_file(src, "jobs/j1/bundle/x.bin")
    store.get_file("jobs/j1/bundle/x.bin", tmp_path / "out" / "x.bin")
    assert (tmp_path / "out" / "x.bin").read_bytes() == b"file data"
    with pytest.raises(StorageNotFoundError):
        store.get_file("jobs/j1/missing", tmp_path / "missing")
    store.delete("jobs/j1/status.json")
    store.delete("jobs/j1/status.json")  # missing: not an error
    assert store.get_bytes("jobs/j1/status.json") is None


def test_list_prefix_semantics(store: GoogleDriveStorage) -> None:
    keys = [
        "jobs/j1/checkpoints/ckpt-1/meta.json",
        "jobs/j1/checkpoints/ckpt-2/meta.json",
        "jobs/j1/checkpoints/LATEST.json",
        "jobs/j1/status.json",
        "jobs/j10/status.json",
        "data-cache/abc/x.bin",
    ]
    for key in keys:
        store.put_bytes(b"x", key)
    assert store.list("") == sorted(keys)
    assert store.list("jobs/j1/") == sorted(k for k in keys if k.startswith("jobs/j1/"))
    assert store.list("jobs/j1") == sorted(k for k in keys if k.startswith("jobs/j1"))
    assert store.list("jobs/j1/checkpoints/ckpt-") == keys[:2]
    assert store.list("missing/") == []
    # A fresh instance (no folder cache) sees the same.
    again = GoogleDriveStorage("root-folder", store.api, retry=FAST)
    assert again.list("jobs/") == sorted(k for k in keys if k.startswith("jobs/"))


def test_overwrite_replaces_content_of_the_same_file(
    store: GoogleDriveStorage, drive: FakeDrive
) -> None:
    store.put_bytes(b"v1", "jobs/j1/status.json")
    [file_id] = [e["id"] for e in drive.entries.values() if e["name"] == "status.json"]
    store.put_bytes(b"v2", "jobs/j1/status.json")
    assert [e["id"] for e in drive.entries.values() if e["name"] == "status.json"] == [file_id]
    assert store.get_bytes("jobs/j1/status.json") == b"v2"


def test_duplicate_names_newest_wins_and_are_cleaned(
    store: GoogleDriveStorage, drive: FakeDrive
) -> None:
    store.put_bytes(b"old", "jobs/j1/LATEST.json")
    parent = next(e["parent"] for e in drive.entries.values() if e["name"] == "LATEST.json")
    drive.upload(parent, "LATEST.json", b"newer (concurrent create)", None)
    assert store.get_bytes("jobs/j1/LATEST.json") == b"newer (concurrent create)"
    store.put_bytes(b"final", "jobs/j1/LATEST.json")
    assert [e["data"] for e in drive.entries.values() if e["name"] == "LATEST.json"] == [b"final"]


def test_rate_limits_are_retried_and_not_found_is_not(
    store: GoogleDriveStorage, drive: FakeDrive
) -> None:
    store.put_bytes(b"x", "jobs/j1/a.json")
    drive.fail_next = [429, 503]
    assert store.get_bytes("jobs/j1/a.json") == b"x"
    drive.fail_next = [404]
    calls = drive.calls
    assert store.list("jobs/") == [] and drive.calls == calls + 1  # 404: no retry


def test_sha256_from_drive(store: GoogleDriveStorage) -> None:
    store.put_bytes(b"payload", "jobs/j1/blob")
    assert store.sha256("jobs/j1/blob") == hashlib.sha256(b"payload").hexdigest()
    assert store.sha256("jobs/j1/none") is None


def test_checkpoint_round_trip(store: GoogleDriveStorage, tmp_path: Path) -> None:
    mgr = CheckpointManager(
        store, job_id="job-g", segment_id="seg-1", local_root=tmp_path / "l", retry=FAST
    )
    mgr.snapshot({"step": 7, "losses": [1.0, 0.5]}, safe_point_id="sp")
    result = mgr.close()
    assert result is not None and result.ok, result
    loaded = load_latest_checkpoint(store, "job-g", tmp_path / "restore", retry=FAST)
    assert loaded is not None and loaded.seq == 1
    ns: dict[str, Any] = {}
    loaded.restore(ns)
    assert ns == {"step": 7, "losses": [1.0, 0.5]}


def test_uri_and_setup(
    store: GoogleDriveStorage, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert storage_uri(store) == "gdrive:root-folder"
    monkeypatch.delenv("GPURELAY_GDRIVE_CREDENTIALS", raising=False)
    monkeypatch.setattr("gpurelay.credentials.get_secret", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="init --gdrive"):
        open_storage("gdrive:root-folder")
    config = Config()
    assert setup_gdrive(config, drive) == "root-folder"  # reuses the existing store folder
    assert config.storage.kind == "gdrive" and config.storage.gdrive_folder_id == "root-folder"
    fresh = FakeDrive()
    new_id = setup_gdrive(Config(), fresh)
    assert fresh.entries[new_id]["name"] == "gpurelay-store"
