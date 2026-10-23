"""Two-phase checkpoint: commit protocol, retention, fallback, and fault injection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from fakes import FaultyStorage, SimulatedKill
from gpurelay.errors import CheckpointCorruptError
from gpurelay.retry import RetryPolicy
from gpurelay.runtime import checkpoint as ckpt_mod
from gpurelay.runtime.checkpoint import (
    CheckpointManager,
    LoadedCheckpoint,
    load_latest_checkpoint,
    read_latest_meta,
    read_pointer,
)
from gpurelay.storage import LocalFSStorage, Storage
from gpurelay.storage.layout import checkpoints_prefix, ckpt_prefix, latest_key, manifest_key

JOB = "job-test"
FAST_RETRY = RetryPolicy(attempts=4, base_delay_s=0.0, jitter=0.0, sleep=lambda _s: None)
TESTS_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_store(tmp_path: Path) -> LocalFSStorage:
    return LocalFSStorage(tmp_path / "store")


@pytest.fixture
def faulty(local_store: LocalFSStorage) -> FaultyStorage:
    return FaultyStorage(local_store)


def _manager(storage: Storage, tmp_path: Path, segment: str = "seg-1") -> CheckpointManager:
    return CheckpointManager(
        storage,
        job_id=JOB,
        segment_id=segment,
        local_root=tmp_path / "local" / segment,
        retry=FAST_RETRY,
    )


def _commit(mgr: CheckpointManager, ns: dict[str, Any], sp: str = "sp") -> int:
    info = mgr.snapshot(ns, safe_point_id=sp, positions={"step": ns.get("step")})
    result = mgr.wait()
    assert result is not None and result.ok, result
    return info.seq


def _load(storage: Storage, tmp_path: Path) -> LoadedCheckpoint | None:
    return load_latest_checkpoint(storage, JOB, tmp_path / "restore", retry=FAST_RETRY)


def _ckpt_seqs(storage: Storage) -> list[int]:
    seqs = {
        int(k.split("/")[3].removeprefix("ckpt-"))
        for k in storage.list(checkpoints_prefix(JOB) + "/ckpt-")
    }
    return sorted(seqs)


# ---------------------------------------------------------------- happy path


def test_round_trip_and_layout(local_store: LocalFSStorage, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "log.txt").write_text("hello\n")
    mgr = _manager(local_store, tmp_path)
    ns = {"step": 3, "losses": [1.0, 0.5, 0.25]}
    info = mgr.snapshot(ns, safe_point_id="L1:3", positions={"L1": 3}, extra={"outputs": outputs})
    assert info.seq == 1 and info.t_snap_s >= 0 and info.size_bytes > 0
    result = mgr.wait()
    assert result is not None and result.ok
    assert mgr.last_committed_seq == 1
    mgr.close()

    assert local_store.list(ckpt_prefix(JOB, 1) + "/") == [
        f"{ckpt_prefix(JOB, 1)}/{name}"
        for name in [
            "manifest.json",
            "meta.json",
            "outputs/log.txt.zst",
            "rng.pt.zst",
            "vars.pkl.zst",
        ]
    ]
    pointer = read_pointer(local_store, JOB)
    assert pointer is not None and pointer.seq == 1 and pointer.safe_point_id == "L1:3"
    assert local_store.get_bytes(manifest_key(JOB, 1)) == local_store.get_bytes(latest_key(JOB))
    assert not list((tmp_path / "local" / "seg-1").iterdir()), "local snapshot not cleaned"

    loaded = _load(local_store, tmp_path)
    assert loaded is not None and loaded.seq == 1 and not loaded.fell_back
    assert loaded.positions == {"L1": 3}
    assert loaded.meta["segment_id"] == "seg-1"
    assert read_latest_meta(local_store, JOB) == loaded.meta  # meta only, no state download
    assert set(loaded.meta["env"]) >= {"python", "platform", "libraries"}
    assert (loaded.local_dir / "outputs" / "log.txt").read_text() == "hello\n"
    fresh: dict[str, Any] = {"losses": []}
    loaded.restore(fresh)
    assert fresh == ns


def test_nothing_committed_returns_none(local_store: LocalFSStorage, tmp_path: Path) -> None:
    assert _load(local_store, tmp_path) is None
    assert read_latest_meta(local_store, JOB) is None


def test_uncommitted_manifest_is_not_a_checkpoint(
    local_store: LocalFSStorage, tmp_path: Path
) -> None:
    mgr = _manager(local_store, tmp_path)
    _commit(mgr, {"step": 1})
    local_store.delete(latest_key(JOB))  # as if the commit write never happened
    assert _load(local_store, tmp_path) is None


def test_seq_continues_across_segments(local_store: LocalFSStorage, tmp_path: Path) -> None:
    first = _manager(local_store, tmp_path, "seg-1")
    _commit(first, {"step": 1})
    _commit(first, {"step": 2})
    first.close()
    second = _manager(local_store, tmp_path, "seg-2")
    assert second.last_committed_seq == 2
    assert _commit(second, {"step": 3}) == 3
    pointer = read_pointer(local_store, JOB)
    assert pointer is not None and pointer.segment_id == "seg-2"


def test_retention_keeps_last_two(local_store: LocalFSStorage, tmp_path: Path) -> None:
    mgr = _manager(local_store, tmp_path)
    for step in range(5):
        _commit(mgr, {"step": step})
    assert _ckpt_seqs(local_store) == [4, 5]


# ---------------------------------------------------------------- persist faults


def test_transient_storage_failures_are_retried(faulty: FaultyStorage, tmp_path: Path) -> None:
    mgr = _manager(faulty, tmp_path)
    faulty.fail_puts = 3
    assert _commit(mgr, {"step": 1}) == 1


def test_corrupted_upload_is_detected_and_reuploaded(faulty: FaultyStorage, tmp_path: Path) -> None:
    mgr = _manager(faulty, tmp_path)
    faulty.corrupt_puts = 1
    assert _commit(mgr, {"step": 1}) == 1
    loaded = _load(faulty, tmp_path)
    assert loaded is not None and loaded.seq == 1


def test_storage_outage_keeps_previous_commit(faulty: FaultyStorage, tmp_path: Path) -> None:
    mgr = _manager(faulty, tmp_path)
    _commit(mgr, {"step": 1})
    faulty.fail_puts = 10_000
    mgr.snapshot({"step": 2}, safe_point_id="sp2")
    result = mgr.wait()
    assert result is not None and not result.ok and isinstance(result.error, OSError)
    assert mgr.last_committed_seq == 1
    assert mgr.stats.failed_persists == 1

    pointer = read_pointer(faulty, JOB)
    assert pointer is not None and pointer.seq == 1
    # Storage recovers: the next checkpoint commits normally.
    faulty.fail_puts = 0
    assert _commit(mgr, {"step": 3}) == 3
    loaded = _load(faulty, tmp_path)
    assert loaded is not None and loaded.seq == 3


def test_kill_during_persist_keeps_previous_commit(faulty: FaultyStorage, tmp_path: Path) -> None:
    mgr = _manager(faulty, tmp_path)
    _commit(mgr, {"step": 1})
    faulty.kill_after_puts = 1  # dies on the second file upload of the next persist
    mgr.snapshot({"step": 2}, safe_point_id="sp2")
    with pytest.raises(SimulatedKill):
        mgr.wait()
    assert faulty.list(ckpt_prefix(JOB, 2) + "/"), "expected a partial upload"

    loaded = _load(faulty, tmp_path)
    assert loaded is not None and loaded.seq == 1
    fresh: dict[str, Any] = {}
    loaded.restore(fresh)
    assert fresh == {"step": 1}

    # Next segment: never reuses seq 2; retention removes the partial upload.
    nxt = _manager(faulty, tmp_path, "seg-2")
    assert nxt.next_seq == 3
    _commit(nxt, {"step": 3})
    assert _ckpt_seqs(faulty) == [1, 3]


def test_kill_during_snapshot_keeps_previous_commit(
    local_store: LocalFSStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = _manager(local_store, tmp_path)
    _commit(mgr, {"step": 1})

    def die(*_args: object, **_kwargs: object) -> None:
        raise SimulatedKill("killed during snapshot")

    monkeypatch.setattr(ckpt_mod, "capture_state", die)
    with pytest.raises(SimulatedKill):
        mgr.snapshot({"step": 2}, safe_point_id="sp2")
    monkeypatch.undo()

    assert not list((tmp_path / "local" / "seg-1").iterdir()), "tmp snapshot left behind"
    assert _ckpt_seqs(local_store) == [1]
    loaded = _load(local_store, tmp_path)
    assert loaded is not None and loaded.seq == 1
    # The same manager keeps working after the failed snapshot.
    assert _commit(mgr, {"step": 3}) == 2


def test_one_persist_at_a_time(faulty: FaultyStorage, tmp_path: Path) -> None:
    faulty.put_delay_s = 0.03
    mgr = _manager(faulty, tmp_path)
    for step in range(3):
        mgr.snapshot({"step": step, "blob": os.urandom(1000)}, safe_point_id=f"sp{step}")
    mgr.close()
    assert faulty.max_active_puts == 1
    assert mgr.stats.total_stall_s > 0
    assert mgr.stats.commits == 3
    pointer = read_pointer(faulty, JOB)
    assert pointer is not None and pointer.seq == 3


# ---------------------------------------------------------------- loader fallback


def _two_commits(storage: Storage, tmp_path: Path) -> None:
    mgr = _manager(storage, tmp_path)
    _commit(mgr, {"step": 1})
    _commit(mgr, {"step": 2})
    mgr.close()


def _restored_step(loaded: LoadedCheckpoint | None) -> Any:
    assert loaded is not None
    ns: dict[str, Any] = {}
    loaded.restore(ns)
    return ns["step"]


def test_corrupted_file_falls_back_to_previous(local_store: LocalFSStorage, tmp_path: Path) -> None:
    _two_commits(local_store, tmp_path)
    local_store.put_bytes(b"bit rot", f"{ckpt_prefix(JOB, 2)}/vars.pkl.zst")
    loaded = _load(local_store, tmp_path)
    assert loaded is not None and loaded.seq == 1 and loaded.fell_back
    assert loaded.problems and "ChecksumMismatchError" in loaded.problems[0]
    assert _restored_step(loaded) == 1


def test_missing_file_falls_back_to_previous(local_store: LocalFSStorage, tmp_path: Path) -> None:
    _two_commits(local_store, tmp_path)
    local_store.delete(f"{ckpt_prefix(JOB, 2)}/rng.pt.zst")
    loaded = _load(local_store, tmp_path)
    assert loaded is not None and loaded.seq == 1
    assert _restored_step(loaded) == 1


def test_corrupted_latest_pointer_uses_manifests(
    local_store: LocalFSStorage, tmp_path: Path
) -> None:
    _two_commits(local_store, tmp_path)
    local_store.put_bytes(b"{not json", latest_key(JOB))
    loaded = _load(local_store, tmp_path)
    assert loaded is not None and loaded.seq == 2 and loaded.fell_back
    assert _restored_step(loaded) == 2


def test_all_checkpoints_corrupted_raises(local_store: LocalFSStorage, tmp_path: Path) -> None:
    _two_commits(local_store, tmp_path)
    for seq in (1, 2):
        local_store.put_bytes(b"bit rot", f"{ckpt_prefix(JOB, seq)}/vars.pkl.zst")
    with pytest.raises(CheckpointCorruptError, match="no checkpoint verifies"):
        _load(local_store, tmp_path)


# ---------------------------------------------------------------- hard kills (real processes)

_KILL_SCRIPT = """
import os, sys
sys.path.insert(0, {tests_dir!r})
from pathlib import Path
from fakes import FaultyStorage
from gpurelay.runtime.checkpoint import CheckpointManager
from gpurelay.storage import LocalFSStorage

class DieOnPickle:
    def __reduce__(self):
        os._exit(9)

store = FaultyStorage(LocalFSStorage(Path({store!r})))
mgr = CheckpointManager(store, job_id={job!r}, segment_id="seg-1", local_root=Path({local!r}))
ns = {{"step": 1, "blob": os.urandom(200_000)}}
mgr.snapshot(ns, safe_point_id="sp1")
assert mgr.wait().ok
ns["step"] = 2
if {mode!r} == "persist":
    store.exit_after_puts = 1      # os._exit on the 2nd upload of the next persist
else:
    ns["poison"] = DieOnPickle()   # os._exit while the snapshot is being written
mgr.snapshot(ns, safe_point_id="sp2")
mgr.wait()
print("UNREACHABLE")
"""


@pytest.mark.parametrize("mode", ["persist", "snapshot"])
def test_hard_kill_process_keeps_previous_commit(
    local_store: LocalFSStorage, tmp_path: Path, mode: str
) -> None:
    local = tmp_path / "local" / "seg-1"
    script = textwrap.dedent(_KILL_SCRIPT).format(
        tests_dir=str(TESTS_DIR), store=str(local_store.root), job=JOB, local=str(local), mode=mode
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 9, proc.stderr
    assert "UNREACHABLE" not in proc.stdout

    pointer = read_pointer(local_store, JOB)
    assert pointer is not None and pointer.seq == 1
    assert _restored_step(_load(local_store, tmp_path)) == 1

    # A new segment on the same machine cleans stale local snapshots and continues.
    nxt = _manager(local_store, tmp_path, "seg-1")
    assert not list(local.iterdir())
    assert _commit(nxt, {"step": 3}) >= 2
    assert _restored_step(_load(local_store, tmp_path)) == 3


def test_meta_json_is_plain_json(local_store: LocalFSStorage, tmp_path: Path) -> None:
    mgr = _manager(local_store, tmp_path)
    _commit(mgr, {"step": 1})
    meta = json.loads(local_store.get_bytes(f"{ckpt_prefix(JOB, 1)}/meta.json") or b"")
    assert meta["seq"] == 1 and meta["job_id"] == JOB
    assert meta["state"]["plain"] == ["step"]
