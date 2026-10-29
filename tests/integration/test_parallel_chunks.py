"""Phase 9 item 1: relay.chunks(..., parallel=True) spread over several backends.

A job runs on the real controller with three Simulated backends: the main session walks
the chunks in order while worker segments compute chunks from the end. The merged log,
outputs, and results must equal a sequential run, also when sessions are killed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from equivalence import CORPUS, RunOutput, assert_equivalent, run_job

from gpurelay.config import (
    BackendsSettings,
    Config,
    LocalSettings,
    NotifySettings,
    RuntimeSettings,
    SchedulerSettings,
    SimulatedSettings,
    StorageSettings,
)
from gpurelay.controller.service import open_context, submit_job
from gpurelay.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

RUNTIME = RuntimeSettings(
    min_interval_s=0.3, max_lost_work_s=1.0, safety_margin_s=0.3, heartbeat_s=0.5,
    stale_heartbeat_s=4, replay_threshold_s=0.5,
)  # fmt: skip
ENV = {"CORPUS_ITEMS": "320", "CORPUS_CHUNK": "20", "CORPUS_CHUNK_SLEEP": "0.8"}
SCRIPT = CORPUS / "parallel_chunks.py"


def _config(home: Path, kill: tuple[float, float] | None) -> Config:
    return Config(
        runtime=RUNTIME,
        scheduler=SchedulerSettings(
            min_segment_h=0.0002, max_infra_retries=8, poll_interval_s=1, max_parallel_workers=2
        ),
        storage=StorageSettings(kind="localfs", localfs_root=home / "store"),
        notify=NotifySettings(desktop=False),
        backends=BackendsSettings(
            local=LocalSettings(enabled=False),
            simulated=[
                SimulatedSettings(name="sim-a", budget_hours=1.0),
                SimulatedSettings(name="sim-b", budget_hours=1.0, kill_between_s=kill, seed=3),
                SimulatedSettings(name="sim-c", budget_hours=1.0),
            ],
        ),
    )


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@pytest.mark.parametrize("kill", [None, (3.0, 5.0)], ids=["clean", "worker-backend-kills"])
def test_parallel_chunks_match_a_sequential_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kill: tuple[float, float] | None
) -> None:
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    home = tmp_path / "home"
    ctx = open_context(_config(home, kill), home)
    job = submit_job(ctx, SCRIPT)
    ctx.controller.run(interval_s=0.3, until_idle=True, max_ticks=3000)
    assert ctx.db.get_job(job.id).state is JobState.COMPLETED  # type: ignore[union-attr]

    segments = ctx.db.segments(job.id)
    workers = [s for s in segments if s.role == "worker"]
    mains = [s for s in segments if s.role == "main"]
    assert workers, [(s.backend, s.role, s.outcome) for s in segments]
    spans = {s.id: (s.started_at.timestamp(), s.ended_at.timestamp()) for s in segments}  # type: ignore[union-attr]
    assert any(_overlap(spans[w.id], spans[m.id]) for w in workers for m in mains)
    done_dir = home / "store" / "jobs" / job.id / "parallel"
    markers = [json.loads(p.read_text()) for p in done_dir.glob("*/done/*.json")]
    assert len(markers) == 16
    worker_ids = {w.id for w in workers}
    assert any(m["by"] in worker_ids for m in markers), "no chunk was computed by a worker"

    baseline = run_job(SCRIPT, tmp_path / "baseline", env=ENV, settings=RUNTIME)
    assert baseline.outcomes == ["completed"]
    assert_equivalent(baseline, RunOutput(result=baseline.result,
                                          final_dir=home / "outputs" / job.id))  # fmt: skip


def test_plain_python_run_matches_too(tmp_path: Path) -> None:
    """Outside GPU Relay the same per-chunk seeding applies: identical output."""
    env = {**os.environ, **ENV, "CORPUS_CHUNK_SLEEP": "0", "PYTHONHASHSEED": "0"}
    plain = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, env=env,
                           capture_output=True, text=True, check=True)  # fmt: skip
    relayed = run_job(SCRIPT, tmp_path / "relay", env={**ENV, "CORPUS_CHUNK_SLEEP": "0"},
                      settings=RUNTIME)  # fmt: skip
    assert plain.stdout.replace("\r\n", "\n") == relayed.log.replace("\r\n", "\n")
    summary = (tmp_path / "summary.txt").read_text()
    assert summary == (tmp_path / "relay" / "final" / "files" / "summary.txt").read_text()


KILL_RUNTIME = RuntimeSettings(
    min_interval_s=1.5, max_lost_work_s=2.5, safety_margin_s=0.3, heartbeat_s=0.5,
    stale_heartbeat_s=4, replay_threshold_s=0.5,
)  # fmt: skip
KILL_ENV = {"CORPUS_ITEMS": "800", "CORPUS_CHUNK": "20", "CORPUS_CHUNK_SLEEP": "0.3"}


@pytest.mark.parametrize("seed", [0, 3])
def test_chunks_done_after_the_last_checkpoint_keep_their_output(tmp_path: Path, seed: int) -> None:
    # Several chunks finish (and are marked done) between two checkpoints; a kill there must
    # not lose their printed lines: the next session takes those chunks as done and prints
    # their saved output (their lines in the killed session's log are past its checkpoint).
    baseline = run_job(SCRIPT, tmp_path / "a", env=KILL_ENV, settings=KILL_RUNTIME)
    run = run_job(SCRIPT, tmp_path / "b", env=KILL_ENV, settings=KILL_RUNTIME,
                  kill_between_s=(6.0, 9.0), seed=seed)  # fmt: skip
    assert run.outcomes.count("crashed") >= 1, run.outcomes
    assert_equivalent(baseline, run)
