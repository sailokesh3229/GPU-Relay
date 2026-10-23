"""Stress (hardening campaign): many jobs, random kills, parallel chunks, controller restarts.

Twelve jobs of four kinds run on three Simulated backends (one kills sessions at random)
while the controller itself is restarted twice. Every job must complete with outputs equal
to an uninterrupted run, and no job may ever have two live main segments.
"""

from __future__ import annotations

import itertools
import time
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
from gpurelay.controller.service import Context, open_context, submit_job
from gpurelay.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

RUNTIME = RuntimeSettings(
    min_interval_s=0.3, max_lost_work_s=1.0, safety_margin_s=0.3, heartbeat_s=0.5,
    stale_heartbeat_s=4, replay_threshold_s=0.5,
)  # fmt: skip
ENV = {"CORPUS_ITEMS": "1800", "CORPUS_CHUNK_SLEEP": "0.1", "CORPUS_CHUNK": "60"}
KINDS = ["chunked_eval.py", "sdk_chunks.py", "chunked_eval_notebook.ipynb", "parallel_chunks.py"]


def _config(home: Path) -> Config:
    return Config(
        runtime=RUNTIME,
        scheduler=SchedulerSettings(
            min_segment_h=0.0002, max_infra_retries=10, poll_interval_s=1, max_parallel_workers=1
        ),
        storage=StorageSettings(kind="localfs", localfs_root=home / "store"),
        notify=NotifySettings(desktop=False),
        backends=BackendsSettings(
            local=LocalSettings(enabled=False),
            simulated=[
                SimulatedSettings(
                    name="sim-killer", budget_hours=5.0, kill_between_s=(2.5, 4.0), seed=11
                ),
                SimulatedSettings(name="sim-short", budget_hours=5.0, max_session_s=4.0),
                SimulatedSettings(name="sim-plain", budget_hours=5.0),
            ],
        ),
    )


def _drive(ctx: Context, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and ctx.controller.busy():
        ctx.controller.tick()
        time.sleep(0.25)


def test_many_jobs_kills_and_controller_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    home = tmp_path / "home"
    ctx = open_context(_config(home), home)
    jobs = [submit_job(ctx, CORPUS / KINDS[i % len(KINDS)], name=f"j{i}") for i in range(12)]

    _drive(ctx, 15)
    ctx = open_context(_config(home), home)  # controller restart 1
    _drive(ctx, 15)
    ctx = open_context(_config(home), home)  # controller restart 2
    _drive(ctx, 600)

    final = {j.name: ctx.db.get_job(j.id) for j in jobs}
    failed = {
        name: (KINDS[int(name[1:]) % len(KINDS)], job.state.value, job.message,
               [(s.role, s.backend, s.outcome, s.message) for s in ctx.db.segments(job.id)])
        for name, job in final.items()
        if job is not None and job.state is not JobState.COMPLETED
    }  # fmt: skip
    assert not failed, failed

    segments = ctx.db.segments()
    outcomes = [s.outcome for s in segments]
    assert "crashed" in outcomes and "handoff" in outcomes, outcomes
    for job in jobs:  # never two live main segments of one job
        spans = sorted(
            (s.started_at.timestamp(), s.ended_at.timestamp())  # type: ignore[union-attr]
            for s in segments
            if s.job_id == job.id and s.role == "main" and s.started_at and s.ended_at
        )
        for (_s1, e1), (s2, _e2) in itertools.pairwise(spans):
            assert s2 >= e1 - 1e-6, job.name

    baselines: dict[str, RunOutput] = {}
    for job in jobs:
        kind = KINDS[int(job.name[1:]) % len(KINDS)]
        if kind not in baselines:
            baselines[kind] = run_job(CORPUS / kind, tmp_path / "base" / kind, env=ENV,
                                      settings=RUNTIME)  # fmt: skip
        base = baselines[kind]
        assert_equivalent(base, RunOutput(result=base.result,
                                          final_dir=home / "outputs" / job.id))  # fmt: skip
