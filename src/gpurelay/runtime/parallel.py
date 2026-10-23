"""Parallel chunks regions: ``relay.chunks(..., parallel=True)`` across backends (Phase 9).

The **main** session runs the job as usual and walks the chunks in order. **Worker** segments
(launched by the controller on spare backends) resume from a main checkpoint taken *inside*
the region and compute chunks from the other end. Coordination goes through storage only::

    jobs/<job>/parallel/<region>/region.json      where the loop is (for workers)
    jobs/<job>/parallel/<region>/claims/<k>.json  "chunk k is being computed by <segment>"
    jobs/<job>/parallel/<region>/done/<k>.json    result + printed output of chunk k

Correctness never depends on claims being exclusive: two sessions computing the same chunk
produce the same result (each chunk is seeded from its id), so a lost race only wastes work.
A claim counts as live while its content keeps changing (its owner refreshes it on every
heartbeat), judged on the *reader's* clock, so clock skew between machines does not matter.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gpurelay.storage.base import BatchStorage, Storage
from gpurelay.storage.layout import blob_key, parallel_prefix, worker_key


def region_id(bucket: str) -> str:
    """Storage-safe id of a chunks loop (its results bucket: position key or name)."""
    return hashlib.sha1(bucket.encode("utf-8")).hexdigest()[:16]


def read_assignment(storage: Storage, job_id: str, segment_id: str) -> str | None:
    """Region a worker segment was launched for, or None for a main segment."""
    raw = storage.get_bytes(worker_key(job_id, segment_id))
    return str(json.loads(raw)["region"]) if raw else None


def write_assignment(storage: Storage, job_id: str, segment_id: str, region: str) -> None:
    storage.put_bytes(json.dumps({"region": region}).encode(), worker_key(job_id, segment_id))


def inside_region(position: dict[str, Any] | None, info: dict[str, Any]) -> bool:
    """Is a checkpoint ``position`` inside the region's loop (an iteration of it)?"""
    if not position or position.get("stmt") != info["stmt"]:
        return False
    path = [list(step) for step in position.get("path", [])]
    address = [list(step) for step in info["address"]]
    depth = len(address)
    return len(path) > depth and path[:depth] == address and path[depth][0] == info["event"]


@dataclass
class ParallelRegion:
    """Shared state of one parallel chunks region, as seen by one session."""

    storage: Storage
    job_id: str
    region: str
    segment_id: str
    stale_s: float
    clock: Callable[[], float] = time.monotonic
    _seen: dict[int, tuple[bytes, float]] = field(default_factory=dict)
    _refresh: int = 0

    @property
    def prefix(self) -> str:
        return parallel_prefix(self.job_id, self.region)

    def _claim_key(self, k: int) -> str:
        return f"{self.prefix}/claims/{k}.json"

    def _done_key(self, k: int) -> str:
        return f"{self.prefix}/done/{k}.json"

    # ------------------------------------------------------------------ region

    def write_info(self, info: dict[str, Any]) -> None:
        self.storage.put_bytes(json.dumps(info).encode(), f"{self.prefix}/region.json")

    def read_info(self) -> dict[str, Any] | None:
        raw = self.storage.get_bytes(f"{self.prefix}/region.json")
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------ chunks

    def done(self, k: int) -> dict[str, Any] | None:
        raw = self.storage.get_bytes(self._done_key(k))
        return json.loads(raw) if raw else None

    def done_count(self) -> int:
        return sum(1 for key in self.storage.list(f"{self.prefix}/done/") if key.endswith(".json"))

    def claimed_by_other(self, k: int) -> bool:
        """Is chunk ``k`` being computed by another live session right now?"""
        raw = self.storage.get_bytes(self._claim_key(k))
        if raw is None:
            self._seen.pop(k, None)
            return False
        if json.loads(raw).get("by") == self.segment_id:
            return False
        now = self.clock()
        seen = self._seen.get(k)
        if seen is None or seen[0] != raw:
            self._seen[k] = (raw, now)  # changed (or first seen): its owner is alive
            return True
        return now - seen[1] < self.stale_s

    def claim(self, k: int) -> None:
        self._refresh += 1
        body = {"by": self.segment_id, "refresh": self._refresh, "at": time.time()}
        self.storage.put_bytes(json.dumps(body).encode(), self._claim_key(k))

    def mark_done(self, k: int, result: tuple[str, bytes], output: tuple[str, bytes]) -> None:
        """Publish chunk ``k``: blobs first, then the marker (one commit when possible).

        ``result``/``output`` are (sha256, bytes); an empty sha means "none".
        """
        marker = {"by": self.segment_id, "result": result[0] or None, "output": output[0] or None}
        items = [(blob_key(self.job_id, sha), data) for sha, data in (result, output) if sha]
        items.append((self._done_key(k), json.dumps(marker).encode()))
        if isinstance(self.storage, BatchStorage):
            self.storage.put_many(items, message=f"gpurelay {self.job_id} chunk {k}")
        else:
            for key, data in items:
                self.storage.put_bytes(data, key)
        self.storage.delete(self._claim_key(k))
