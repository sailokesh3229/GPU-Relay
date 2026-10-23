"""Two-phase checkpointing (spec §8): snapshot (blocking) then persist (background).

Local layout of a snapshot (uncompressed, before persist)::

    <local_root>/ckpt-<seq>.tmp/   while being written
    <local_root>/ckpt-<seq>/       complete; handed to the persist thread
        meta.json  rng.pt  torch_state.pt  vars.pkl  <extra entries...>

Remote layout after commit (spec §11.3)::

    jobs/<job>/checkpoints/ckpt-<seq>/meta.json, manifest.json, *.zst
    jobs/<job>/checkpoints/LATEST.json        # commit pointer
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import zstandard
from pydantic import ValidationError

from gpurelay.constants import DEFAULT_KEEP_CHECKPOINTS, DEFAULT_ZSTD_LEVEL
from gpurelay.errors import CheckpointCorruptError, ChecksumMismatchError, StorageError
from gpurelay.models import CheckpointPointer, utcnow
from gpurelay.retry import RetryPolicy
from gpurelay.runtime.env import env_fingerprint
from gpurelay.runtime.persist import (
    ZSTD_SUFFIX,
    Persister,
    PersistResult,
    decompress_file,
)
from gpurelay.runtime.rng import load_rng, save_rng
from gpurelay.runtime.state import RestoreReport, StateManifest, capture_state, restore_state
from gpurelay.storage.base import Storage, sha256_file
from gpurelay.storage.layout import (
    MANIFEST_FILENAME,
    checkpoints_prefix,
    ckpt_dirname,
    ckpt_prefix,
    latest_key,
    manifest_key,
    parse_ckpt_seq,
)

log = logging.getLogger(__name__)

META_FILE = "meta.json"
RNG_FILE = "rng.pt"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class SnapshotInfo:
    """Measurements of one snapshot (inputs to adaptive frequency, spec §8.2)."""

    seq: int
    local_dir: Path
    t_snap_s: float
    stall_s: float
    size_bytes: int
    state: StateManifest


@dataclass
class CheckpointStats:
    snapshots: int = 0
    commits: int = 0
    failed_persists: int = 0
    total_stall_s: float = 0.0
    last_snapshot: SnapshotInfo | None = None
    last_persist: PersistResult | None = None


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _copy_into(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


# ====================================================================== writer


class CheckpointManager:
    """Takes snapshots at safe points and persists them in the background.

    At most one persist runs at a time: a snapshot requested while a persist is in
    flight waits for it, and the wait is recorded as stall time (spec §8.1 step 5).
    """

    def __init__(
        self,
        storage: Storage,
        *,
        job_id: str,
        segment_id: str,
        local_root: Path,
        zstd_level: int = DEFAULT_ZSTD_LEVEL,
        keep: int = DEFAULT_KEEP_CHECKPOINTS,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.storage = storage
        self.job_id = job_id
        self.segment_id = segment_id
        self.retry = retry or RetryPolicy()
        self.local_root = Path(local_root)
        self._reset_local_root()

        latest = read_pointer(storage, job_id)
        self.last_committed_seq: int | None = latest.seq if latest else None
        self._next_seq = self._discover_next_seq(latest)
        self._persister = Persister(
            storage,
            job_id,
            zstd_level=zstd_level,
            keep=keep,
            retry=self.retry,
            committed=[latest.seq] if latest else [],
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpurelay-persist")
        self._inflight: Future[PersistResult] | None = None
        self.stats = CheckpointStats()

    # ------------------------------------------------------------------ setup

    def _reset_local_root(self) -> None:
        """Remove snapshots left behind by a previous (killed) process on this machine."""
        if self.local_root.exists():
            for entry in self.local_root.iterdir():
                if entry.name.startswith("ckpt-"):
                    shutil.rmtree(entry, ignore_errors=True)
        self.local_root.mkdir(parents=True, exist_ok=True)

    def _discover_next_seq(self, latest: CheckpointPointer | None) -> int:
        """Never reuse a seq seen in storage, committed or not."""
        seen = [latest.seq] if latest else []
        keys = self.retry.call(
            lambda: self.storage.list(checkpoints_prefix(self.job_id) + "/"), what="list"
        )
        seen.extend(s for k in keys if (s := parse_ckpt_seq(k, self.job_id)) is not None)
        return max(seen, default=0) + 1

    # ------------------------------------------------------------------ API

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def mark_blobs_known(self, shas: Iterable[str]) -> None:
        """Blobs already committed in storage (skip existence checks on upload)."""
        self._persister.known_blobs.update(shas)

    def snapshot(
        self,
        namespace: Mapping[str, Any],
        *,
        safe_point_id: str,
        names: Iterable[str] | None = None,
        positions: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        extra: Mapping[str, Path] | None = None,
        uploads: Mapping[str, Path] | None = None,
        blobs: Mapping[str, Path] | None = None,
    ) -> SnapshotInfo:
        """Capture a snapshot at a safe point and start persisting it in the background.

        ``positions`` (resume position) and ``runtime`` (output chain, file manifest,
        ...) are JSON-serializable and stored in ``meta.json``. ``extra`` maps names
        inside the checkpoint to local files/directories to include. ``uploads`` maps
        remote keys to local files that are copied now and uploaded before the commit.
        ``blobs`` maps sha256 -> immutable local file for the content-addressed blob store.
        """
        stall_started = time.perf_counter()
        self._wait_inflight()
        stall_s = time.perf_counter() - stall_started

        seq = self._next_seq
        tmp = self.local_root / f"{ckpt_dirname(seq)}.tmp"
        final = self.local_root / ckpt_dirname(seq)
        uploads_dir = self.local_root / f"{ckpt_dirname(seq)}.uploads"
        created_at = utcnow()
        started = time.perf_counter()
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            save_rng(tmp / RNG_FILE)
            manifest = capture_state(namespace, tmp, names)
            for rel, src in (extra or {}).items():
                _copy_into(Path(src), tmp / rel)
            shutil.rmtree(uploads_dir, ignore_errors=True)
            upload_copies: dict[str, Path] = {}
            for index, (remote, src) in enumerate(sorted((uploads or {}).items())):
                upload_copies[remote] = uploads_dir / str(index)
                _copy_into(Path(src), upload_copies[remote])
            meta = {
                "format": CHECKPOINT_FORMAT_VERSION,
                "seq": seq,
                "job_id": self.job_id,
                "segment_id": self.segment_id,
                "safe_point_id": safe_point_id,
                "created_at": created_at.isoformat(),
                "positions": dict(positions or {}),
                "runtime": dict(runtime or {}),
                "state": manifest.model_dump(mode="json"),
                "env": env_fingerprint(),
            }
            (tmp / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
            shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(uploads_dir, ignore_errors=True)
            raise
        t_snap_s = time.perf_counter() - started
        self._next_seq += 1

        info = SnapshotInfo(
            seq=seq,
            local_dir=final,
            t_snap_s=t_snap_s,
            stall_s=stall_s,
            size_bytes=_dir_size(final),
            state=manifest,
        )
        self.stats.snapshots += 1
        self.stats.total_stall_s += stall_s
        self.stats.last_snapshot = info
        self._inflight = self._executor.submit(
            self._persist_and_cleanup,
            seq,
            final,
            uploads_dir,
            safe_point_id,
            created_at,
            upload_copies,
            dict(blobs or {}),
        )
        return info

    def _persist_and_cleanup(
        self,
        seq: int,
        final: Path,
        uploads_dir: Path,
        safe_point_id: str,
        created_at: datetime,
        uploads: dict[str, Path],
        blobs: dict[str, Path],
    ) -> PersistResult:
        try:
            return self._persister.persist(
                seq,
                final,
                segment_id=self.segment_id,
                safe_point_id=safe_point_id,
                created_at=created_at,
                uploads=uploads,
                blobs=blobs,
            )
        finally:
            shutil.rmtree(uploads_dir, ignore_errors=True)

    def wait(self) -> PersistResult | None:
        """Block until the in-flight persist (if any) finishes and return its result."""
        return self._wait_inflight()

    def close(self) -> PersistResult | None:
        """Wait for the in-flight persist and stop the background worker."""
        try:
            return self._wait_inflight()
        finally:
            self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------ internals

    def _wait_inflight(self) -> PersistResult | None:
        future, self._inflight = self._inflight, None
        if future is None:
            return None
        result = future.result()  # re-raises BaseException (e.g. a simulated kill)
        self.stats.last_persist = result
        if result.ok:
            self.stats.commits += 1
            self.last_committed_seq = result.seq
        else:
            self.stats.failed_persists += 1
        return result


# ====================================================================== reader


@dataclass
class LoadedCheckpoint:
    """A committed checkpoint downloaded, verified, and decompressed locally."""

    pointer: CheckpointPointer
    local_dir: Path
    meta: dict[str, Any]
    fell_back: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def seq(self) -> int:
        return self.pointer.seq

    @property
    def positions(self) -> dict[str, Any]:
        positions: dict[str, Any] = self.meta.get("positions", {})
        return positions

    @property
    def runtime(self) -> dict[str, Any]:
        runtime: dict[str, Any] = self.meta.get("runtime", {})
        return runtime

    def restore(self, namespace: MutableMapping[str, Any]) -> RestoreReport:
        """Restore variables into ``namespace`` (live objects first), then RNG state."""
        report = restore_state(namespace, self.local_dir)
        report.warnings.extend(load_rng(self.local_dir / RNG_FILE))
        return report


def _parse_pointer(raw: bytes | None, what: str) -> CheckpointPointer | None:
    if raw is None:
        return None
    try:
        return CheckpointPointer.model_validate_json(raw)
    except ValidationError as exc:
        log.warning("%s is unreadable: %s", what, exc)
        return None


def read_pointer(storage: Storage, job_id: str) -> CheckpointPointer | None:
    """Return the committed ``LATEST.json`` pointer, or None if missing/unreadable."""
    return _parse_pointer(storage.get_bytes(latest_key(job_id)), "LATEST.json")


def read_latest_meta(storage: Storage, job_id: str) -> dict[str, Any] | None:
    """``meta.json`` of the committed checkpoint, without downloading its state.

    Returns None if nothing is committed or the meta cannot be read and verified.
    """
    pointer = read_pointer(storage, job_id)
    if pointer is None:
        return None
    name = next((n for n in pointer.files if n in (META_FILE, META_FILE + ZSTD_SUFFIX)), None)
    if name is None:
        return None
    raw = storage.get_bytes(f"{ckpt_prefix(job_id, pointer.seq)}/{name}")
    if raw is None or hashlib.sha256(raw).hexdigest() != pointer.sha256s.get(name):
        return None
    if name.endswith(ZSTD_SUFFIX):
        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
    meta: dict[str, Any] = json.loads(raw)
    return meta if meta.get("seq") == pointer.seq else None


def _candidates(storage: Storage, job_id: str) -> list[CheckpointPointer]:
    """Committed checkpoints to try, newest first: LATEST, then older manifests."""
    latest_raw = storage.get_bytes(latest_key(job_id))
    if latest_raw is None:
        return []  # nothing was ever committed (spec §8.1: no LATEST, no checkpoint)
    latest = _parse_pointer(latest_raw, "LATEST.json")
    candidates = [latest] if latest else []
    limit = latest.seq if latest else None

    manifest_seqs = sorted(
        {
            seq
            for key in storage.list(checkpoints_prefix(job_id) + "/")
            if key.endswith("/" + MANIFEST_FILENAME)
            and (seq := parse_ckpt_seq(key, job_id)) is not None
            and (limit is None or seq < limit)
        },
        reverse=True,
    )
    for seq in manifest_seqs:
        pointer = _parse_pointer(storage.get_bytes(manifest_key(job_id, seq)), f"ckpt-{seq}")
        if pointer is not None and pointer.seq == seq:
            candidates.append(pointer)
    return candidates


def _download_verified(
    storage: Storage, job_id: str, pointer: CheckpointPointer, dest: Path, retry: RetryPolicy
) -> dict[str, Any]:
    """Download, verify, and decompress one checkpoint into ``dest``; return meta."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    prefix = ckpt_prefix(job_id, pointer.seq)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp_name:
        tmp = Path(tmp_name)
        for name in pointer.files:
            downloaded = tmp / name
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            retry.call(
                partial(storage.get_file, f"{prefix}/{name}", downloaded),
                what=f"download {name}",
            )
            actual = sha256_file(downloaded)
            if actual != pointer.sha256s[name]:
                raise ChecksumMismatchError(f"{name}: sha256 {actual} != {pointer.sha256s[name]}")
            if name.endswith(ZSTD_SUFFIX):
                out = dest / name[: -len(ZSTD_SUFFIX)]
                out.parent.mkdir(parents=True, exist_ok=True)
                decompress_file(downloaded, out)
            else:
                out = dest / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(downloaded, out)
    meta: dict[str, Any] = json.loads((dest / META_FILE).read_text(encoding="utf-8"))
    if meta.get("seq") != pointer.seq or meta.get("job_id") != job_id:
        raise ChecksumMismatchError(f"meta.json does not match checkpoint {pointer.seq}")
    return meta


def load_latest_checkpoint(
    storage: Storage, job_id: str, dest_root: Path, *, retry: RetryPolicy | None = None
) -> LoadedCheckpoint | None:
    """Load the newest committed checkpoint that verifies, falling back to older ones.

    Returns None when nothing was ever committed. Raises
    :class:`CheckpointCorruptError` when checkpoints were committed but none verifies.
    """
    retry = retry or RetryPolicy()
    candidates = retry.call(lambda: _candidates(storage, job_id), what="read checkpoints")
    if not candidates and storage.exists(latest_key(job_id)):
        raise CheckpointCorruptError(f"{job_id}: LATEST.json is unreadable and no fallback")
    latest_seq = read_pointer(storage, job_id)
    problems: list[str] = []
    for pointer in candidates:
        dest = Path(dest_root) / ckpt_dirname(pointer.seq)
        try:
            meta = _download_verified(storage, job_id, pointer, dest, retry)
        except (StorageError, OSError, ValueError, KeyError) as exc:
            message = f"checkpoint {pointer.seq} unusable: {type(exc).__name__}: {exc}"
            log.warning(message)
            problems.append(message)
            shutil.rmtree(dest, ignore_errors=True)
            continue
        if problems:
            log.warning("fell back to checkpoint %d", pointer.seq)
        return LoadedCheckpoint(
            pointer=pointer,
            local_dir=dest,
            meta=meta,
            fell_back=latest_seq is None or pointer.seq != latest_seq.seq,
            problems=problems,
        )
    if candidates:
        raise CheckpointCorruptError(f"{job_id}: no checkpoint verifies: " + "; ".join(problems))
    return None
