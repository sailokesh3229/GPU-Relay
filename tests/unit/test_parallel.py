"""Parallel chunks regions (Phase 9 item 1): claims, markers, seeding, controller workers."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest
from test_controller import Harness, _harness, stub_assembly  # noqa: F401 (autouse)

import gpurelay as relay
from gpurelay.models import JobState, ParallelStatus, RuntimeState, SegmentState, StatusFile
from gpurelay.runtime.parallel import ParallelRegion, inside_region, read_assignment, region_id
from gpurelay.runtime.rng import chunk_seed
from gpurelay.storage import LocalFSStorage
from gpurelay.storage.layout import blob_key, status_key

# ---------------------------------------------------------------- shared region state


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_claims_are_live_while_they_change(tmp_path: Path) -> None:
    storage = LocalFSStorage(tmp_path)
    clock = Clock()
    owner = ParallelRegion(storage, "job-1", "r1", "seg-a", stale_s=10, clock=clock)
    other = ParallelRegion(storage, "job-1", "r1", "seg-b", stale_s=10, clock=clock)
    assert not other.claimed_by_other(3)
    owner.claim(3)
    assert not owner.claimed_by_other(3)  # its own claim
    assert other.claimed_by_other(3)  # first seen: alive
    clock.now += 9
    assert other.claimed_by_other(3)
    clock.now += 2  # unchanged for 11 s: the owner is gone
    assert not other.claimed_by_other(3)
    owner.claim(3)  # heartbeat refresh
    assert other.claimed_by_other(3)


def test_mark_done_publishes_blobs_then_marker(tmp_path: Path) -> None:
    storage = LocalFSStorage(tmp_path)
    region = ParallelRegion(storage, "job-1", "r1", "seg-a", stale_s=10)
    region.claim(0)
    region.mark_done(0, ("aa" * 32, b"result"), ("", b""))
    assert region.done(0) == {"by": "seg-a", "result": "aa" * 32, "output": None}
    assert storage.get_bytes(blob_key("job-1", "aa" * 32)) == b"result"
    assert not storage.exists(f"{region.prefix}/claims/0.json")
    assert region.done_count() == 1 and region.done(1) is None


def test_inside_region() -> None:
    info = {"stmt": 4, "address": [], "event": 1}
    assert inside_region({"stmt": 4, "path": [[1, 7, None]], "end": "iter"}, info)
    assert not inside_region({"stmt": 4, "path": [[0, 7, None]], "end": "iter"}, info)
    assert not inside_region({"stmt": 5, "path": [[1, 7, None]], "end": "iter"}, info)
    assert not inside_region({"stmt": 4, "end": "stmt"}, info)
    nested = {"stmt": 2, "address": [[0, 3, "L"]], "event": 0}
    assert inside_region({"stmt": 2, "path": [[0, 3, "L"], [0, 5, None]], "end": "iter"}, nested)
    assert not inside_region({"stmt": 2, "path": [[0, 4, "L"], [0, 5, None]]}, nested)


# ---------------------------------------------------------------- seeding contract


def test_parallel_chunks_outside_relay_are_seeded_per_chunk() -> None:
    def run() -> tuple[list[float], float]:
        random.seed(123)
        for chunk in relay.chunks(range(10), 3, parallel=True, seed=5):
            relay.save_result(chunk.id, random.random())
        return relay.load_results(), random.random()

    first, after = run()
    random.seed(chunk_seed(5, 2))
    assert first[2] == random.random()  # chunk 2 depends only on (seed, id)
    random.seed(chunk_seed(5, 4))
    assert after == random.random()  # 4 chunks: the RNG ends seeded with chunk_seed(5, 4)
    assert run() == (first, after)


def test_parallel_chunks_need_a_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        relay.chunks(iter(range(5)), 2, parallel=True)


# ---------------------------------------------------------------- controller


def _region_status(h: Harness, job_id: str, *, total: int, done: int, ready: bool) -> None:
    [seg] = [s for s in h.live(job_id) if s.role == "main"]
    status = StatusFile(
        job_id=job_id, segment_id=seg.id, state=RuntimeState.RUNNING,
        heartbeat_at=datetime.fromtimestamp(h.clock.now, timezone.utc),
        parallel=ParallelStatus(region="r1", total=total, done=done, ready=ready),
    )  # fmt: skip
    h.storage.put_bytes(status.model_dump_json().encode(), status_key(job_id))


def test_workers_launch_on_spare_backends_once_the_region_is_ready(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4"), "b": (None, True, 1, "T4"),
                            "c": (None, True, 1, "T4")})  # fmt: skip
    job = h.submit()
    h.controller.tick()
    _region_status(h, job.id, total=10, done=1, ready=False)
    h.controller.tick()
    assert not [s for s in h.live(job.id) if s.role == "worker"]
    _region_status(h, job.id, total=10, done=1, ready=True)
    h.controller.tick()
    workers = [s for s in h.live(job.id) if s.role == "worker"]
    assert len(workers) == 2 and {w.region for w in workers} == {"r1"}
    assert len({w.backend for w in workers} | {h.live(job.id)[0].backend}) == 3
    assert read_assignment(h.storage, job.id, workers[0].id) == "r1"
    h.controller.tick()
    assert len([s for s in h.live(job.id) if s.role == "worker"]) == 2  # no more than the limit


def test_workers_never_change_the_job_and_stop_with_it(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4"), "b": (None, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    _region_status(h, job.id, total=3, done=1, ready=True)
    h.controller.tick()
    [worker] = [s for s in h.live(job.id) if s.role == "worker"]
    h.backends[worker.backend].finish(job.id, worker.id, 87, "failed_infra")
    h.controller.tick()  # a failing worker: no infra failure counted, job keeps running
    assert h.job(job.id).state is JobState.RUNNING and h.job(job.id).infra_failures == 0
    _region_status(h, job.id, total=3, done=1, ready=True)
    h.controller.tick()  # helps again
    [worker] = [s for s in h.live(job.id) if s.role == "worker"]
    main = next(s for s in h.live(job.id) if s.role == "main")
    h.backends[main.backend].finish(job.id, main.id, 0, "completed")
    h.controller.tick()
    assert h.job(job.id).state is JobState.COMPLETED
    stopped = next(s for s in h.db.segments(job.id) if s.id == worker.id)
    assert stopped.state is not SegmentState.RUNNING and stopped.outcome == "cancelled"


def test_region_ids_are_storage_safe() -> None:
    rid = region_id("S12/L0[3]/L1")
    assert len(rid) == 16 and rid.isalnum()
    assert json.dumps(rid)
