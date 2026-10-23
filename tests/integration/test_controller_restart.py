"""Controller restart mid-job (plan Phase 8): state is recovered from SQLite + storage.

* The session survives the restart: the new controller (which never launched it) follows it
  through ``status.json`` to completion.
* The session dies with the controller (e.g. the machine rebooted): the new controller sees
  the heartbeat stop, relaunches, and the job resumes from its last checkpoint.

Both must end with outputs equivalent to an uninterrupted run.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from equivalence import CORPUS, RunOutput, assert_equivalent, run_job

from gpurelay.config import (
    BackendsSettings,
    Config,
    LocalSettings,
    NotifySettings,
    RuntimeSettings,
    SchedulerSettings,
    StorageSettings,
)
from gpurelay.controller.service import Context, open_context, submit_job
from gpurelay.models import JobState
from gpurelay.runtime.checkpoint import read_pointer

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

RUNTIME = RuntimeSettings(
    min_interval_s=0.2, max_lost_work_s=0.6, safety_margin_s=0.3, heartbeat_s=0.5,
    stale_heartbeat_s=4, replay_threshold_s=0.5,
)  # fmt: skip
ENV = {"CORPUS_ITEMS": "400", "CORPUS_CHUNK_SLEEP": "0.1"}
SCRIPT = "chunked_eval.py"


def _config(home: Path) -> Config:
    return Config(
        runtime=RUNTIME,
        scheduler=SchedulerSettings(min_segment_h=0.0002, poll_interval_s=1),
        storage=StorageSettings(kind="localfs", localfs_root=home / "store"),
        notify=NotifySettings(desktop=False),
        backends=BackendsSettings(local=LocalSettings(enabled=True)),
    )


def _tick_until(ctx: Context, done: Any, timeout_s: float = 180) -> None:
    end = time.monotonic() + timeout_s
    while not done():
        assert time.monotonic() < end, [
            (s.backend, s.state.value, s.outcome, s.message) for s in ctx.db.segments()
        ]
        ctx.controller.tick()
        time.sleep(0.3)


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("session_survives", [True, False], ids=["survives", "dies"])
def test_controller_restart_mid_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_survives: bool
) -> None:
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    home = tmp_path / "home"
    first = open_context(_config(home), home)
    job = submit_job(first, CORPUS / SCRIPT)
    _tick_until(first, lambda: read_pointer(first.storage, job.id) is not None)
    [seg] = first.db.segments(job.id)
    assert first.db.get_job(job.id).state is JobState.RUNNING  # type: ignore[union-attr]
    if not session_survives:
        _kill_tree(int(seg.handle_extra["pid"]))
    # The first controller is never used again: only SQLite and storage carry over.

    second = open_context(_config(home), home)
    second.controller.run(interval_s=0.3, until_idle=True, max_ticks=2000)
    assert second.db.get_job(job.id).state is JobState.COMPLETED  # type: ignore[union-attr]
    segments = second.db.segments(job.id)
    if session_survives:
        assert [s.outcome for s in segments] == ["completed"]
    else:
        assert segments[0].outcome == "crashed"
        assert segments[-1].outcome == "completed" and len(segments) >= 2

    baseline = run_job(CORPUS / SCRIPT, tmp_path / "baseline", env=ENV, settings=RUNTIME)
    assert_equivalent(baseline, RunOutput(result=baseline.result,
                                          final_dir=home / "outputs" / job.id))  # fmt: skip
