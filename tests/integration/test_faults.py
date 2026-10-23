"""Whole-job fault injection (hardening campaign).

* Flaky storage: every storage call inside the sessions fails transiently with some
  probability, while sessions are also killed. The job must still finish identically.
* A corrupted latest checkpoint between sessions: the next session falls back to the
  previous checkpoint and the job still finishes identically.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from equivalence import CORPUS, RunOutput, assert_equivalent, run_job

from gpurelay.backends.simulated import SimulatedBackend
from gpurelay.config import RuntimeSettings
from gpurelay.controller.local_runner import SegmentRecord, run_local_job
from gpurelay.models import JobState
from gpurelay.runtime.checkpoint import read_pointer
from gpurelay.storage import LocalFSStorage
from gpurelay.storage.layout import ckpt_prefix

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETTINGS = RuntimeSettings(
    min_interval_s=0.3, max_lost_work_s=1.0, safety_margin_s=0.3, heartbeat_s=1,
    replay_threshold_s=0.5,
)  # fmt: skip
ENV = {"CORPUS_ITEMS": "800", "CORPUS_CHUNK_SLEEP": "0.1"}
SITE = Path(__file__).resolve().parent / "faulty_site"


def test_flaky_storage_and_kills(tmp_path: Path) -> None:
    script = CORPUS / "chunked_eval.py"
    baseline = run_job(script, tmp_path / "a", env=ENV, settings=SETTINGS)
    flaky = {**ENV, "PYTHONPATH": str(SITE), "GPURELAY_TEST_FAULT_RATE": "0.05"}
    run = run_job(script, tmp_path / "b", env=flaky, settings=SETTINGS,
                  kill_between_s=(4.0, 6.0), seed=2)  # fmt: skip
    assert "crashed" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)


def test_corrupted_latest_checkpoint_falls_back(tmp_path: Path) -> None:
    script = CORPUS / "chunked_eval.py"
    baseline = run_job(script, tmp_path / "a", env=ENV, settings=SETTINGS)
    store = LocalFSStorage(tmp_path / "b" / "store")
    backend = SimulatedBackend(store, tmp_path / "b" / "sessions", settings=SETTINGS,
                               max_session_s=3.0, env=ENV)  # fmt: skip
    corrupted: list[int] = []

    def corrupt_newest(record: SegmentRecord) -> None:
        # After the first handoff, damage the newest checkpoint as a bad disk or a partial
        # upload would: the next segment must fall back to the previous one.
        pointer = read_pointer(store, job_id)
        if corrupted or record.outcome != "handoff" or pointer is None or pointer.seq < 2:
            return
        for key in store.list(ckpt_prefix(job_id, pointer.seq) + "/"):
            if key.endswith("vars.pkl.zst"):
                store.put_bytes(b"garbage", key)
                corrupted.append(pointer.seq)

    job_id = "job-corrupt"
    result = run_local_job(script, storage=store, backend=backend, out_dir=tmp_path / "b" / "final",
                           job_id=job_id, on_segment=corrupt_newest)  # fmt: skip
    assert corrupted, [r.outcome for r in result.segments]
    assert result.state is JobState.COMPLETED, (result.state, result.message)
    final = read_pointer(store, job_id)
    assert final is not None and final.seq > corrupted[0]  # numbering moved on, no overwrite
    assert_equivalent(baseline, RunOutput(result=result, final_dir=tmp_path / "b" / "final"))
