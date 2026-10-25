"""Persist phase of the two-phase checkpoint (spec §8.1).

Runs in a background thread:

1. compress every snapshot file with zstd (JSON files are uploaded as-is);
2. upload each file to ``jobs/<job>/checkpoints/ckpt-<seq>/`` with retry/backoff;
3. verify the stored SHA-256 of every file;
4. write ``ckpt-<seq>/manifest.json`` and then **commit** by replacing ``LATEST.json``;
5. delete checkpoints older than the newest ``keep`` committed ones.

A failure (or a kill) anywhere before step 4 leaves ``LATEST.json`` pointing at the
previous committed checkpoint.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import IO, BinaryIO, cast

import zstandard

from gpurelay.errors import ChecksumMismatchError
from gpurelay.models import CheckpointPointer
from gpurelay.retry import RetryPolicy
from gpurelay.storage.base import BatchStorage, ChecksumStorage, Storage, sha256_file
from gpurelay.storage.layout import (
    MANIFEST_FILENAME,
    blob_key,
    checkpoints_prefix,
    ckpt_prefix,
    latest_key,
    manifest_key,
    parse_ckpt_seq,
)

log = logging.getLogger(__name__)

ZSTD_SUFFIX = ".zst"
_UNCOMPRESSED_SUFFIXES = (".json",)


@dataclass
class PersistResult:
    """Outcome and measurements of one persist (used for adaptive frequency, spec §8.2)."""

    seq: int
    ok: bool
    error: Exception | None = None
    raw_bytes: int = 0
    stored_bytes: int = 0
    compress_s: float = 0.0
    upload_s: float = 0.0
    total_s: float = 0.0
    files: list[str] = field(default_factory=list)
    retention_done: bool = False  # old checkpoints were deleted in the data commit itself

    @property
    def throughput_bps(self) -> float:
        """Raw (uncompressed) bytes persisted per second, end to end."""
        return self.raw_bytes / self.total_s if self.total_s > 0 else 0.0


class _HashingWriter:
    """File-like sink that hashes everything written through it."""

    def __init__(self, fh: BinaryIO) -> None:
        self._fh = fh
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return self._fh.write(data)


def compress_file(src: Path, dst: Path, level: int) -> str:
    """zstd-compress ``src`` into ``dst``; return the SHA-256 of the compressed bytes."""
    cctx = zstandard.ZstdCompressor(level=level)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        sink = _HashingWriter(fout)
        cctx.copy_stream(fin, cast("IO[bytes]", sink))
    return sink.digest.hexdigest()


def decompress_file(src: Path, dst: Path) -> None:
    """Decompress a zstd file."""
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        dctx.copy_stream(fin, fout)


def stored_sha256(storage: Storage, remote: str, scratch_dir: Path) -> str | None:
    """SHA-256 of a stored object: cheap if the storage supports it, else download."""
    if isinstance(storage, ChecksumStorage):
        return storage.sha256(remote)
    if not storage.exists(remote):
        return None
    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
        local = Path(tmp) / "verify.bin"
        storage.get_file(remote, local)
        return sha256_file(local)


class Persister:
    """Uploads local snapshots and commits them. Not thread-safe: one persist at a time."""

    def __init__(
        self,
        storage: Storage,
        job_id: str,
        *,
        zstd_level: int,
        keep: int,
        retry: RetryPolicy,
        committed: list[int] | None = None,
    ) -> None:
        if keep < 2:
            raise ValueError("keep must be >= 2 so a fallback checkpoint always exists")
        self.storage = storage
        self.job_id = job_id
        self.zstd_level = zstd_level
        self.keep = keep
        self.retry = retry
        self.committed: list[int] = list(committed or [])
        self.known_blobs: set[str] = set()  # blob shas known to exist in storage

    def persist(
        self,
        seq: int,
        local_dir: Path,
        *,
        segment_id: str,
        safe_point_id: str,
        created_at: datetime,
        uploads: Mapping[str, Path] | None = None,
        blobs: Mapping[str, Path] | None = None,
    ) -> PersistResult:
        """Persist ``local_dir`` as checkpoint ``seq``. Never raises ``Exception``.

        Before the commit, also uploads ``uploads`` (remote key -> local file, always
        overwritten) and ``blobs`` (sha256 -> local file, stored content-addressed at
        ``jobs/<job>/blobs/<sha256>`` and skipped when already present).
        """
        result = PersistResult(seq=seq, ok=False)
        started = time.perf_counter()
        staging = local_dir.with_name(local_dir.name + ".persist")
        try:
            if isinstance(self.storage, BatchStorage):
                self._persist_batched(
                    seq, local_dir, staging, result, segment_id, safe_point_id, created_at,
                    dict(uploads or {}), dict(blobs or {}),
                )  # fmt: skip
            else:
                self._upload_blobs(blobs or {}, local_dir.parent, result)
                for remote, path in (uploads or {}).items():
                    digest = sha256_file(path)
                    self.retry.call(
                        partial(self._upload_verified, path, remote, digest, local_dir.parent),
                        what=f"upload {remote}",
                    )
                    result.stored_bytes += path.stat().st_size
                self._persist(
                    seq, local_dir, staging, result, segment_id, safe_point_id, created_at
                )
            result.ok = True
        except Exception as exc:
            result.error = exc
            log.error("persist of checkpoint %d failed: %s: %s", seq, type(exc).__name__, exc)
        finally:
            result.total_s = time.perf_counter() - started
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(local_dir, ignore_errors=True)
        if result.ok and not result.retention_done:
            self._apply_retention(seq)
        return result

    # ------------------------------------------------------------------ steps

    def _persist(
        self,
        seq: int,
        local_dir: Path,
        staging: Path,
        result: PersistResult,
        segment_id: str,
        safe_point_id: str,
        created_at: datetime,
    ) -> None:
        prefix = ckpt_prefix(self.job_id, seq)
        # Leftovers from an earlier uncommitted attempt at this seq must not linger.
        for key in self.retry.call(lambda: self.storage.list(prefix + "/"), what="list"):
            self.retry.call(partial(self.storage.delete, key), what=f"delete {key}")

        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        sha256s: dict[str, str] = {}
        for src in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            rel = src.relative_to(local_dir).as_posix()
            result.raw_bytes += src.stat().st_size
            t0 = time.perf_counter()
            if rel.endswith(_UNCOMPRESSED_SUFFIXES):
                upload, name, digest = src, rel, sha256_file(src)
            else:
                name = rel + ZSTD_SUFFIX
                upload = staging / name
                upload.parent.mkdir(parents=True, exist_ok=True)
                digest = compress_file(src, upload, self.zstd_level)
            t1 = time.perf_counter()
            self.retry.call(
                partial(self._upload_verified, upload, f"{prefix}/{name}", digest, staging),
                what=f"upload {name}",
            )
            result.compress_s += t1 - t0
            result.upload_s += time.perf_counter() - t1
            result.stored_bytes += upload.stat().st_size
            sha256s[name] = digest

        pointer = CheckpointPointer(
            seq=seq,
            files=sorted(sha256s),
            sha256s=sha256s,
            safe_point_id=safe_point_id,
            created_at=created_at,
            segment_id=segment_id,
        )
        body = pointer.model_dump_json(indent=2).encode()
        self.retry.call(
            lambda: self.storage.put_bytes(body, manifest_key(self.job_id, seq)),
            what="write manifest",
        )
        # Commit point: the checkpoint exists from here on.
        self.retry.call(
            lambda: self.storage.put_bytes(body, latest_key(self.job_id)), what="commit LATEST"
        )
        self.committed.append(seq)
        result.files = pointer.files
        log.info("committed checkpoint %d (%d files)", seq, len(pointer.files))

    def _stage(
        self, local_dir: Path, staging: Path, result: PersistResult
    ) -> list[tuple[str, Path, str]]:
        """Compress snapshot files; return (name in checkpoint, file to upload, sha256)."""
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        staged = []
        for src in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            rel = src.relative_to(local_dir).as_posix()
            result.raw_bytes += src.stat().st_size
            t0 = time.perf_counter()
            if rel.endswith(_UNCOMPRESSED_SUFFIXES):
                staged.append((rel, src, sha256_file(src)))
            else:
                name = rel + ZSTD_SUFFIX
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                staged.append((name, target, compress_file(src, target, self.zstd_level)))
            result.compress_s += time.perf_counter() - t0
        return staged

    def _persist_batched(
        self,
        seq: int,
        local_dir: Path,
        staging: Path,
        result: PersistResult,
        segment_id: str,
        safe_point_id: str,
        created_at: datetime,
        uploads: dict[str, Path],
        blobs: dict[str, Path],
    ) -> None:
        """One write for blobs + uploads + checkpoint files + manifest + the deletion of old
        checkpoints, then the commit.

        On the Hub this is two commits per checkpoint (data, then ``LATEST.json``): the Hub
        allows 128 commits per hour per repo.
        """
        assert isinstance(self.storage, BatchStorage)
        batch = self.storage
        prefix = ckpt_prefix(self.job_id, seq)
        # Leftovers of an uncommitted attempt at this seq go in the same commit.
        stale = self.retry.call(lambda: self.storage.list(prefix + "/"), what="list")
        # Retention too. ``LATEST`` still names the newest committed checkpoint until the
        # second commit, so that one is always kept (``keep >= 2``).
        try:
            previous = sorted(set(self.committed))[-(self.keep - 1) :]
            doomed = self._doomed(seq, set(previous))
            result.retention_done = True
        except Exception as exc:  # listing failed: retention runs after the commit instead
            log.warning("could not plan checkpoint retention: %s", exc)
            doomed = []

        expected: dict[str, str] = {}
        items: list[tuple[str, Path | bytes]] = []
        for digest, path in sorted(blobs.items()):
            remote = blob_key(self.job_id, digest)
            if digest in self.known_blobs or self.retry.call(
                partial(self.storage.exists, remote), what="exists"
            ):
                self.known_blobs.add(digest)
                continue
            items.append((remote, path))
            expected[remote] = digest
        for remote, path in sorted(uploads.items()):
            items.append((remote, path))
            expected[remote] = sha256_file(path)
        sha256s: dict[str, str] = {}
        for name, path, digest in self._stage(local_dir, staging, result):
            items.append((f"{prefix}/{name}", path))
            expected[f"{prefix}/{name}"] = digest
            sha256s[name] = digest
        pointer = CheckpointPointer(
            seq=seq,
            files=sorted(sha256s),
            sha256s=sha256s,
            safe_point_id=safe_point_id,
            created_at=created_at,
            segment_id=segment_id,
        )
        body = pointer.model_dump_json(indent=2).encode()
        items.append((manifest_key(self.job_id, seq), body))
        expected[manifest_key(self.job_id, seq)] = hashlib.sha256(body).hexdigest()
        result.stored_bytes += sum(
            len(src) if isinstance(src, bytes) else src.stat().st_size for _r, src in items
        )

        def write_and_verify() -> None:
            batch.put_many(items, message=f"gpurelay {self.job_id} checkpoint {seq}",
                           deletes=[*stale, *doomed])  # fmt: skip
            for remote, digest in expected.items():
                stored = stored_sha256(self.storage, remote, staging)
                if stored != digest:
                    raise ChecksumMismatchError(f"{remote}: stored sha256 {stored} != {digest}")

        started = time.perf_counter()
        self.retry.call(write_and_verify, what=f"upload checkpoint {seq}")
        result.upload_s += time.perf_counter() - started
        self.known_blobs.update(blobs)
        # Commit point: the checkpoint exists from here on.
        self.retry.call(
            lambda: self.storage.put_bytes(body, latest_key(self.job_id)), what="commit LATEST"
        )
        self.committed.append(seq)
        result.files = pointer.files
        log.info("committed checkpoint %d (%d files, batched)", seq, len(pointer.files))

    def _upload_blobs(
        self, blobs: Mapping[str, Path], scratch: Path, result: PersistResult
    ) -> None:
        for digest, path in sorted(blobs.items()):
            if digest in self.known_blobs:
                continue
            remote = blob_key(self.job_id, digest)
            exists = self.retry.call(partial(self.storage.exists, remote), what="exists")
            if not exists:
                self.retry.call(
                    partial(self._upload_verified, path, remote, digest, scratch),
                    what=f"upload blob {digest[:12]}",
                )
                result.stored_bytes += path.stat().st_size
            self.known_blobs.add(digest)

    def _upload_verified(self, local: Path, remote: str, digest: str, scratch: Path) -> None:
        self.storage.put_file(local, remote)
        stored = stored_sha256(self.storage, remote, scratch)
        if stored != digest:
            raise ChecksumMismatchError(f"{remote}: stored sha256 {stored} != {digest}")

    def _doomed(self, seq: int, keep: set[int]) -> list[str]:
        """Keys of checkpoints older than ``seq`` that are not in ``keep``, manifests first."""
        keys = self.storage.list(checkpoints_prefix(self.job_id) + "/")
        # Manifests first: a partially deleted checkpoint must not look loadable.
        keys.sort(key=lambda k: not k.endswith("/" + MANIFEST_FILENAME))
        return [
            key
            for key in keys
            if (key_seq := parse_ckpt_seq(key, self.job_id)) is not None
            and key_seq < seq
            and key_seq not in keep
        ]

    def _apply_retention(self, seq: int) -> None:
        """Delete every checkpoint older than ``seq`` except the newest ``keep`` committed."""
        keep = set(sorted(set(self.committed))[-self.keep :])
        try:
            doomed = self._doomed(seq, keep)
            if isinstance(self.storage, BatchStorage):
                manifests = [k for k in doomed if k.endswith("/" + MANIFEST_FILENAME)]
                self.storage.delete_many(manifests, message=f"gpurelay {self.job_id} retention")
                rest = [k for k in doomed if k not in manifests]
                self.storage.delete_many(rest, message=f"gpurelay {self.job_id} retention")
            else:
                for key in doomed:
                    self.storage.delete(key)
        except Exception as exc:  # retention is best effort; the commit already happened
            log.warning("checkpoint retention failed: %s", exc)
