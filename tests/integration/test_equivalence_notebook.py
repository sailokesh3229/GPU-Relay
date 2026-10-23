"""Phase 4 verification: unmodified notebooks survive kills with identical outputs,
including the assembled ``output.ipynb`` (execution counts aside)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from equivalence import CORPUS, assert_equivalent, notebook_outputs, run_job

from gpurelay.config import RuntimeSettings

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]
pytest.importorskip("IPython")

SETTINGS = RuntimeSettings(
    min_interval_s=0.3,
    max_lost_work_s=1.0,
    safety_margin_s=0.3,
    heartbeat_s=1,
    replay_threshold_s=0.5,
)


def _segment_logs(run_root: Path) -> list[str]:
    logs = sorted(
        (run_root / "store").glob("jobs/*/segments/*/log.txt"), key=lambda p: p.stat().st_mtime
    )
    return [p.read_text(encoding="utf-8") for p in logs]


def test_chunked_notebook_random_kills(tmp_path: Path) -> None:
    nb = CORPUS / "chunked_eval_notebook.ipynb"
    env = {"CORPUS_ITEMS": "800", "CORPUS_CHUNK_SLEEP": "0.1"}
    baseline = run_job(nb, tmp_path / "a", env=env, settings=SETTINGS)
    run = run_job(nb, tmp_path / "b", env=env, settings=SETTINGS, kill_between_s=(2.0, 3.5), seed=1)
    assert run.outcomes.count("crashed") >= 1, run.outcomes
    assert_equivalent(baseline, run)
    cells = notebook_outputs(run)
    assert cells is not None
    assert [o["output_type"] for o in cells[4]] == ["display_data", "execute_result"]


@pytest.mark.torch
def test_mlp_notebook_resumes_inside_the_training_cell(tmp_path: Path) -> None:
    nb = CORPUS / "mlp_notebook.ipynb"
    env = {"CORPUS_EPOCHS": "40", "CORPUS_EPOCH_SLEEP": "0.25"}
    baseline = run_job(nb, tmp_path / "a", env=env, settings=SETTINGS)
    run = run_job(
        nb, tmp_path / "b", env=env, settings=SETTINGS, kill_between_s=(10.0, 14.0), seed=3
    )
    assert "crashed" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)
    # The segment after the kill continued mid-loop, not from the cell's first epoch.
    resumed = [log for log in _segment_logs(tmp_path / "b")[1:] if log]
    assert resumed and resumed[0].startswith("epoch") and not resumed[0].startswith("epoch   0")
    if importlib.util.find_spec("matplotlib") is not None:
        assert len(run.figures()) == 1


@pytest.mark.torch
def test_vae_fid_notebook_random_kills(tmp_path: Path) -> None:
    nb = CORPUS / "vae_fid_notebook.ipynb"
    env = {"CORPUS_EPOCHS": "16", "CORPUS_EPOCH_SLEEP": "0.4"}
    baseline = run_job(nb, tmp_path / "a", env=env, settings=SETTINGS)
    run = run_job(
        nb, tmp_path / "b", env=env, settings=SETTINGS, kill_between_s=(9.0, 12.0), seed=5
    )
    assert "crashed" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)
