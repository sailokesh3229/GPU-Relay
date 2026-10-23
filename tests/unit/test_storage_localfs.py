from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from gpurelay.constants import TMP_MARKER
from gpurelay.errors import InvalidRemotePathError, StorageNotFoundError
from gpurelay.storage import LocalFSStorage, Storage


@pytest.fixture
def store(tmp_path: Path) -> LocalFSStorage:
    return LocalFSStorage(tmp_path / "store")


def test_satisfies_protocol(store: LocalFSStorage) -> None:
    assert isinstance(store, Storage)


def test_put_get_bytes(store: LocalFSStorage) -> None:
    assert store.get_bytes("jobs/j1/status.json") is None
    store.put_bytes(b'{"a": 1}', "jobs/j1/status.json")
    assert store.get_bytes("jobs/j1/status.json") == b'{"a": 1}'
    store.put_bytes(b"replaced", "jobs/j1/status.json")
    assert store.get_bytes("jobs/j1/status.json") == b"replaced"


def test_put_get_file(store: LocalFSStorage, tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(100_000))
    store.put_file(src, "jobs/j1/bundle/bundle.tar.zst")

    dest = tmp_path / "out" / "nested" / "bundle.tar.zst"
    store.get_file("jobs/j1/bundle/bundle.tar.zst", dest)
    assert dest.read_bytes() == src.read_bytes()


def test_get_file_missing_raises(store: LocalFSStorage, tmp_path: Path) -> None:
    with pytest.raises(StorageNotFoundError):
        store.get_file("nope/x.bin", tmp_path / "x.bin")
    assert not (tmp_path / "x.bin").exists()


def test_exists_and_delete(store: LocalFSStorage) -> None:
    store.put_bytes(b"x", "a/b.txt")
    assert store.exists("a/b.txt")
    assert not store.exists("a")  # directories are not keys
    store.delete("a/b.txt")
    assert not store.exists("a/b.txt")
    store.delete("a/b.txt")  # idempotent


def test_list_prefix_semantics(store: LocalFSStorage) -> None:
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
    # String-prefix match, not directory match: "jobs/j1" also matches "jobs/j10".
    assert store.list("jobs/j1") == sorted(k for k in keys if k.startswith("jobs/j1"))
    assert store.list("jobs/j1/checkpoints/ckpt-") == [
        "jobs/j1/checkpoints/ckpt-1/meta.json",
        "jobs/j1/checkpoints/ckpt-2/meta.json",
    ]
    assert store.list("missing/") == []


def test_list_hides_temp_files(store: LocalFSStorage) -> None:
    store.put_bytes(b"x", "jobs/j1/status.json")
    (store.root / "jobs" / "j1" / f"status.json{TMP_MARKER}deadbeef").write_bytes(b"partial")
    assert store.list("jobs/") == ["jobs/j1/status.json"]


@pytest.mark.parametrize(
    "key",
    ["", "/abs", "a//b", "a/./b", "../escape", "a/../b", "a\\b", "c:x", f"a{TMP_MARKER}1", "a/b/"],
)
def test_invalid_keys_rejected(store: LocalFSStorage, key: str) -> None:
    with pytest.raises(InvalidRemotePathError):
        store.put_bytes(b"x", key)


def test_failed_write_keeps_old_content(
    store: LocalFSStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.put_bytes(b"old", "LATEST.json")

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        store.put_bytes(b"new", "LATEST.json")
    monkeypatch.undo()

    assert store.get_bytes("LATEST.json") == b"old"
    assert sorted(p.name for p in store.root.iterdir()) == ["LATEST.json"]


@pytest.mark.timeout(60)
def test_reader_never_sees_partial_file(store: LocalFSStorage) -> None:
    """Concurrent reader observes only complete old or new content (spec §11.1)."""
    size = 2 * 1024 * 1024
    payloads = (b"a" * size, b"b" * size)
    store.put_bytes(payloads[0], "status.json")

    done = threading.Event()
    errors: list[BaseException] = []
    bad_reads: list[int] = []
    reads = 0

    def writer() -> None:
        try:
            for i in range(60):
                store.put_bytes(payloads[i % 2], "status.json")
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    def reader() -> None:
        nonlocal reads
        try:
            while not done.is_set():
                data = store.get_bytes("status.json")
                reads += 1
                if data not in payloads:
                    bad_reads.append(len(data) if data is not None else -1)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert not bad_reads, f"partial reads observed (lengths): {bad_reads[:5]}"
    assert reads > 0
    assert store.list("") == ["status.json"]
