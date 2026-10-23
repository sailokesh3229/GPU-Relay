"""HFHubStorage against an in-memory fake of the Hub API (no network)."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("huggingface_hub")
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from gpurelay.errors import StorageNotFoundError
from gpurelay.retry import RetryPolicy
from gpurelay.runtime.checkpoint import CheckpointManager, load_latest_checkpoint
from gpurelay.storage import BatchStorage, ChecksumStorage, Storage
from gpurelay.storage.hf_hub import HFHubStorage


class NotFound(Exception):
    def __init__(self) -> None:
        super().__init__("404")
        self.response = type("R", (), {"status_code": 404})()


class RateLimited(Exception):
    def __init__(self) -> None:
        super().__init__("429 Too Many Requests")
        self.response = type("R", (), {"status_code": 429})()


@dataclass
class RepoFile:
    path: str
    size: int
    lfs: Any = None


@dataclass
class Lfs:
    sha256: str


@dataclass
class FakeHub:
    files: dict[str, bytes] = field(default_factory=dict)
    commits: list[str] = field(default_factory=list)
    fail_next: int = 0  # simulate rate limiting
    private: bool = True

    def repo_info(self, repo_id: str, repo_type: str, **_: Any) -> Any:
        return type("Info", (), {"private": self.private})()

    def _maybe_fail(self) -> None:
        if self.fail_next:
            self.fail_next -= 1
            raise RateLimited()

    def create_commit(
        self, repo_id: str, operations: list[Any], commit_message: str, **_: Any
    ) -> None:
        self._maybe_fail()
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                src = op.path_or_fileobj
                self.files[op.path_in_repo] = (
                    src if isinstance(src, bytes) else Path(src).read_bytes()
                )
            elif isinstance(op, CommitOperationDelete):
                self.files.pop(op.path_in_repo, None)
        self.commits.append(commit_message)

    def list_repo_tree(self, repo_id: str, path_in_repo: str | None = None, **_: Any) -> list[Any]:
        self._maybe_fail()
        prefix = (path_in_repo + "/") if path_in_repo else ""
        if path_in_repo and not any(p.startswith(prefix) for p in self.files):
            raise NotFound()
        return [RepoFile(p, len(d)) for p, d in self.files.items() if p.startswith(prefix)]

    def file_exists(self, repo_id: str, filename: str, **_: Any) -> bool:
        return filename in self.files

    def get_paths_info(self, repo_id: str, paths: list[str], **_: Any) -> list[Any]:
        out = []
        for path in paths:
            if path in self.files:
                data = self.files[path]
                lfs = Lfs(hashlib.sha256(data).hexdigest()) if len(data) > 64 else None
                out.append(RepoFile(path, len(data), lfs))
        return out

    def download(self, *, repo_id: str, filename: str, local_dir: str, **_: Any) -> str:
        self._maybe_fail()
        if filename not in self.files:
            raise NotFound()
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    def super_squash_history(self, repo_id: str, **_: Any) -> None:
        self.commits = ["squashed"]


FAST = RetryPolicy(attempts=4, base_delay_s=0.0, jitter=0.0, sleep=lambda _s: None)


@pytest.fixture
def hub() -> FakeHub:
    return FakeHub()


@pytest.fixture
def store(hub: FakeHub) -> HFHubStorage:
    return HFHubStorage("me/gpurelay-store", "hf_x", api=hub, downloader=hub.download, retry=FAST)


def test_protocols(store: HFHubStorage) -> None:
    assert isinstance(store, Storage)
    assert isinstance(store, BatchStorage)
    assert isinstance(store, ChecksumStorage)


def test_put_get_list_delete(store: HFHubStorage, hub: FakeHub, tmp_path: Path) -> None:
    assert store.get_bytes("jobs/j/status.json") is None
    store.put_bytes(b"{}", "jobs/j/status.json")
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 100)
    store.put_file(src, "jobs/j/blobs/abc")
    assert store.get_bytes("jobs/j/status.json") == b"{}"
    store.get_file("jobs/j/blobs/abc", tmp_path / "out" / "f.bin")
    assert (tmp_path / "out" / "f.bin").read_bytes() == b"x" * 100
    assert store.list("jobs/j/") == ["jobs/j/blobs/abc", "jobs/j/status.json"]
    assert store.list("jobs/j/blobs/a") == ["jobs/j/blobs/abc"]
    assert store.list("missing/") == []
    assert store.exists("jobs/j/status.json")
    store.delete("jobs/j/status.json")
    store.delete("jobs/j/status.json")  # missing: no error, no commit
    assert not store.exists("jobs/j/status.json")
    assert len(hub.commits) == 3
    with pytest.raises(StorageNotFoundError):
        store.get_file("nope", tmp_path / "nope")


def test_batch_is_one_commit(store: HFHubStorage, hub: FakeHub, tmp_path: Path) -> None:
    src = tmp_path / "a"
    src.write_bytes(b"a")
    store.put_many([("k/a", src), ("k/b", b"b"), ("k/c", b"c")], message="batch")
    assert hub.commits == ["batch"]
    store.delete_many(["k/a", "k/b", "k/zzz"])
    assert sorted(hub.files) == ["k/c"] and len(hub.commits) == 2


def test_sha256_from_lfs_or_download(store: HFHubStorage) -> None:
    big, small = b"y" * 1000, b"z"
    store.put_many([("big", big), ("small", small)])
    assert store.sha256("big") == hashlib.sha256(big).hexdigest()
    assert store.sha256("small") == hashlib.sha256(small).hexdigest()
    assert store.sha256("missing") is None


def test_rate_limits_are_retried(store: HFHubStorage, hub: FakeHub) -> None:
    hub.fail_next = 2
    store.put_bytes(b"1", "a")
    assert hub.files["a"] == b"1"


def test_checkpoint_round_trip_uses_two_commits(
    store: HFHubStorage, hub: FakeHub, tmp_path: Path
) -> None:
    mgr = CheckpointManager(
        store, job_id="job-h", segment_id="seg-1", local_root=tmp_path / "l", retry=FAST
    )
    blob = tmp_path / "blob"
    blob.write_bytes(b"figure")
    sha = hashlib.sha256(b"figure").hexdigest()
    log_copy = tmp_path / "log.txt"
    log_copy.write_text("hello\n")
    mgr.snapshot(
        {"step": 7, "losses": [1.0, 0.5]},
        safe_point_id="sp",
        uploads={"jobs/job-h/segments/seg-1/log.txt": log_copy},
        blobs={sha: blob},
    )
    result = mgr.close()
    assert result is not None and result.ok, result
    assert len(hub.commits) == 2  # data + LATEST
    assert "jobs/job-h/checkpoints/LATEST.json" in hub.files
    assert f"jobs/job-h/blobs/{sha}" in hub.files

    loaded = load_latest_checkpoint(store, "job-h", tmp_path / "restore", retry=FAST)
    assert loaded is not None and loaded.seq == 1
    ns: dict[str, Any] = {}
    loaded.restore(ns)
    assert ns == {"step": 7, "losses": [1.0, 0.5]}
    shutil.rmtree(tmp_path / "restore")


def test_used_bytes_and_squash(store: HFHubStorage, hub: FakeHub) -> None:
    store.put_many([("a", b"12"), ("b", b"345")])
    assert store.used_bytes() == 5
    store.squash_history()
    assert hub.commits == ["squashed"]


def test_is_private_reflects_the_repo(hub: FakeHub, store: HFHubStorage) -> None:
    assert store.is_private() is True
    hub.private = False
    assert store.is_private() is False
