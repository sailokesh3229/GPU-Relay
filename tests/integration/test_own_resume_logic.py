"""Jobs with their own resume logic (``range(start_epoch, ...)`` from their own checkpoint
file) lose no iteration when GPU Relay resumes them (found with the InfoVAE notebook)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from equivalence import assert_equivalent, run_job

from gpurelay.config import RuntimeSettings

pytestmark = pytest.mark.integration

FAST = RuntimeSettings(min_interval_s=0.2, max_lost_work_s=0.5, safety_margin_s=0.3, heartbeat_s=1)
SCRIPT = textwrap.dedent("""\
    import json, os, time

    STATE = "own_checkpoint.json"   # the job's own resume file, like checkpoint.pt
    start_epoch, total = 1, 0
    if os.path.exists(STATE):
        with open(STATE) as fh:
            saved = json.load(fh)
        start_epoch, total = saved["epoch"] + 1, saved["total"]
        print("Resuming from epoch:", start_epoch)

    for epoch in range(start_epoch, 13):
        time.sleep(0.35)
        total += epoch
        print("epoch", epoch, "total", total)
        with open(STATE, "w") as fh:
            json.dump({"epoch": epoch, "total": total}, fh)
    print("done", total)
    """)


def test_every_epoch_runs_once_after_resumes(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text(SCRIPT)
    baseline = run_job(script, tmp_path / "base", settings=FAST)
    run = run_job(script, tmp_path / "kills", settings=FAST, kill_between_s=(1.4, 2.2), seed=5)
    assert len(run.outcomes) >= 3, run.outcomes  # resumed several times
    assert_equivalent(baseline, run)
    log = (run.final_dir / "log.txt").read_text(encoding="utf-8")
    assert [f"epoch {e} " in log for e in range(1, 13)] == [True] * 12
    assert "done 78" in log and "Resuming from epoch" not in log
