"""Phase 5 'done when': a multi-job, multi-backend relay works locally end to end.

Three jobs are submitted to the real controller with three Simulated backends (different
budgets and fake GPU labels; one kills sessions, one forces handoffs). All jobs must
complete with outputs equivalent to uninterrupted runs, in parallel, never with two live
segments per job, and with ledger usage equal to the segments' durations.
"""

from __future__ import annotations

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
    min_interval_s=0.2, max_lost_work_s=0.6, safety_margin_s=0.3, heartbeat_s=1,
    replay_threshold_s=0.5,
)  # fmt: skip
ENV = {"CORPUS_ITEMS": "600", "CORPUS_CHUNK_SLEEP": "0.1"}
JOBS = ["chunked_eval.py", "sdk_chunks.py", "chunked_eval_notebook.ipynb"]


def _config(home: Path) -> Config:
    return Config(
        runtime=RUNTIME,
        scheduler=SchedulerSettings(min_segment_h=0.0002, max_infra_retries=5, poll_interval_s=1),
        storage=StorageSettings(kind="localfs", localfs_root=home / "store"),
        notify=NotifySettings(desktop=False),
        backends=BackendsSettings(
            local=LocalSettings(enabled=False),
            simulated=[
                SimulatedSettings(
                    name="sim-killer",
                    gpu_label="SimT4",
                    budget_hours=1.0,
                    kill_between_s=(2.2, 3.5),
                    seed=1,
                ),
                SimulatedSettings(
                    name="sim-short", gpu_label="SimP100", budget_hours=1.0, max_session_s=3.5
                ),
                SimulatedSettings(name="sim-small", gpu_label="SimT4", budget_hours=0.5),
            ],
        ),
    )


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def test_three_jobs_three_backends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    home = tmp_path / "home"
    ctx = open_context(_config(home), home)
    jobs = [submit_job(ctx, CORPUS / name) for name in JOBS]
    assert len({j.bundle_hash for j in jobs}) == 3

    ctx.controller.run(interval_s=0.3, until_idle=True, max_ticks=2500)

    states = {j.name: ctx.db.get_job(j.id).state for j in jobs}  # type: ignore[union-attr]
    assert set(states.values()) == {JobState.COMPLETED}, states

    # Equivalent to uninterrupted runs.
    for job, name in zip(jobs, JOBS, strict=True):
        baseline = run_job(CORPUS / name, tmp_path / "baseline" / job.id, env=ENV, settings=RUNTIME)
        relayed = RunOutput(result=baseline.result, final_dir=home / "outputs" / job.id)
        assert_equivalent(baseline, relayed)

    segments = ctx.db.segments()
    spans = {
        s.id: (s.started_at.timestamp(), s.ended_at.timestamp())  # type: ignore[union-attr]
        for s in segments
    }
    outcomes = [s.outcome for s in segments]
    assert "crashed" in outcomes and "handoff" in outcomes, outcomes
    assert len({s.backend for s in segments}) == 3

    # Never two live segments of one job; segments of different jobs did run in parallel.
    by_job: dict[str, list[str]] = {}
    for s in segments:
        by_job.setdefault(s.job_id, []).append(s.id)
    for ids in by_job.values():
        for i, a in enumerate(ids):
            assert not any(_overlap(spans[a], spans[b]) for b in ids[i + 1 :])
    assert any(
        _overlap(spans[a.id], spans[b.id])
        for a in segments
        for b in segments
        if a.job_id != b.job_id
    )

    # Ledger totals match segment durations per backend.
    for backend in {s.backend for s in segments}:
        ledger = sum(e.hours or 0 for e in ctx.db.ledger(backend) if e.kind == "usage")
        durations = sum(
            (spans[s.id][1] - spans[s.id][0]) / 3600 for s in segments if s.backend == backend
        )
        assert ledger == pytest.approx(durations, rel=1e-6)
