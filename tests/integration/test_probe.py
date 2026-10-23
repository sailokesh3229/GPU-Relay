"""``gpurelay analyze --probe`` (plan Phase 8): real short runs on this machine."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from equivalence import CORPUS

from gpurelay.analyzer.probe import probe

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def test_probe_measures_rate_and_projects_the_total() -> None:
    env = {"CORPUS_ITEMS": "600", "CORPUS_CHUNK_SLEEP": "0.4"}  # 30 chunks x 0.4 s
    result = probe(CORPUS / "chunked_eval.py", seconds=6, env=env)
    assert result.ended == "handed off"
    assert result.total == 30 and result.completed is not None and 0 < result.completed < 30
    assert result.rate_per_h is not None
    assert result.rate_per_h / 3600 == pytest.approx(1 / 0.4, rel=0.35)
    assert result.projected_h is not None and result.startup_s is not None
    actual_s = result.startup_s + 30 * 0.4
    assert result.projected_h * 3600 == pytest.approx(actual_s, rel=0.35)
    assert result.peak_rss_mb is not None and result.peak_rss_mb > 5
    assert result.checkpoint_mb is not None and result.device == "CPU"


def test_probe_of_a_short_job_completes() -> None:
    started = time.monotonic()
    env = {"CORPUS_ITEMS": "100", "CORPUS_CHUNK_SLEEP": "0"}
    result = probe(CORPUS / "chunked_eval.py", seconds=60, env=env)
    assert result.ended == "completed"
    assert result.projected_h is not None and result.projected_h * 3600 <= result.seconds + 1
    assert time.monotonic() - started < 60  # it did not wait for the deadline


def test_probe_reports_a_failing_job(tmp_path: Path) -> None:
    script = tmp_path / "broken.py"
    script.write_text("for i in range(3):\n    print('step', i)\nraise ValueError('boom')\n")
    result = probe(script, seconds=30)
    assert result.ended == "failed (exit 86)"
    assert "boom" in result.log_tail
