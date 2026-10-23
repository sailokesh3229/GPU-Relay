"""Phase 2 verification: SDK corpus items survive kills and handoffs with identical outputs."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from equivalence import CORPUS, RunOutput, assert_equivalent, lost_work_s, run_job

from gpurelay.backends.simulated import SimulatedBackend
from gpurelay.config import RuntimeSettings
from gpurelay.controller.local_runner import run_local_job
from gpurelay.models import JobState
from gpurelay.storage import LocalFSStorage

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

# Small intervals so short simulated sessions commit often.
FAST = RuntimeSettings(min_interval_s=0.2, max_lost_work_s=0.5, safety_margin_s=0.3, heartbeat_s=1)

CHUNKS = CORPUS / "sdk_chunks.py"
CHUNKS_ENV = {"CORPUS_ITEMS": "800", "CORPUS_CHUNK_SLEEP": "0.1"}  # 40 chunks, ~4 s of work


@pytest.fixture(scope="module")
def chunks_baseline(tmp_path_factory: pytest.TempPathFactory) -> RunOutput:
    run = run_job(CHUNKS, tmp_path_factory.mktemp("baseline"), env=CHUNKS_ENV, settings=FAST)
    assert run.outcomes == ["completed"]
    return run


# ---------------------------------------------------------------- chunked evaluation (no torch)


def test_chunks_matches_plain_python_run(chunks_baseline: RunOutput, tmp_path: Path) -> None:
    """The relay output equals running the script directly (outside GPU Relay)."""
    script = tmp_path / CHUNKS.name
    script.write_bytes(CHUNKS.read_bytes())
    env = {**os.environ, "PYTHONHASHSEED": "0", **CHUNKS_ENV, "CORPUS_CHUNK_SLEEP": "0"}
    proc = subprocess.run(
        [sys.executable, script.name], cwd=tmp_path, env=env, capture_output=True, check=True
    )
    assert proc.stdout.replace(b"\r\n", b"\n").decode() == chunks_baseline.log
    assert (tmp_path / "metrics.json").read_bytes() == chunks_baseline.files()["metrics.json"]


def test_chunks_random_hard_kills(chunks_baseline: RunOutput, tmp_path: Path) -> None:
    run = run_job(
        CHUNKS, tmp_path, env=CHUNKS_ENV, settings=FAST, kill_between_s=(1.2, 2.4), seed=1
    )
    assert run.outcomes.count("crashed") >= 2, run.outcomes
    assert_equivalent(chunks_baseline, run)
    # Work lost per kill <= max_lost_work + one safe-point gap + persist/scheduling slack.
    lost = lost_work_s(run)
    assert lost, "expected kills after a commit in the same segment"
    assert max(lost) <= FAST.max_lost_work_s + 0.1 + 1.0, lost


def test_chunks_forced_handoffs(chunks_baseline: RunOutput, tmp_path: Path) -> None:
    run = run_job(CHUNKS, tmp_path, env=CHUNKS_ENV, settings=FAST, deadline_s=2.0)
    assert run.outcomes.count("handoff") >= 2, run.outcomes
    assert_equivalent(chunks_baseline, run)


def test_chunks_kills_and_handoffs(chunks_baseline: RunOutput, tmp_path: Path) -> None:
    run = run_job(
        CHUNKS,
        tmp_path,
        env=CHUNKS_ENV,
        settings=FAST,
        deadline_s=2.2,
        kill_between_s=(1.0, 3.0),
        seed=7,
    )
    assert len(run.outcomes) >= 3, run.outcomes
    assert_equivalent(chunks_baseline, run)


# ---------------------------------------------------------------- PyTorch MLP


@pytest.mark.torch
def test_mlp_kills_and_handoffs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    env = {"CORPUS_EPOCHS": "40", "CORPUS_EPOCH_SLEEP": "0.25"}  # ~10 s of training
    baseline = run_job(CORPUS / "sdk_mlp.py", tmp_path / "a", env=env, settings=FAST)
    # Startup (python + torch import) takes several seconds on Windows; sessions must be
    # longer than that to make progress.
    run = run_job(
        CORPUS / "sdk_mlp.py",
        tmp_path / "b",
        env=env,
        settings=FAST,
        deadline_s=14,
        kill_between_s=(9.0, 16.0),
        seed=3,
    )
    assert len(run.outcomes) >= 2, run.outcomes
    assert_equivalent(baseline, run)
    if importlib.util.find_spec("matplotlib") is not None:
        assert len(run.figures()) == 1


# ---------------------------------------------------------------- failure modes


def _run_script(tmp_path: Path, source: str, deadline_s: float | None = None) -> object:
    script = tmp_path / "job_script.py"
    script.write_text(textwrap.dedent(source))
    storage = LocalFSStorage(tmp_path / "store")
    backend = SimulatedBackend(
        storage, tmp_path / "sessions", settings=FAST, max_session_s=deadline_s or 3600
    )
    console: list[str] = []
    result = run_local_job(
        script,
        storage=storage,
        backend=backend,
        out_dir=tmp_path / "final",
        on_console=console.append,
    )
    return result, "".join(console)


def test_user_error_reports_original_line_and_does_not_retry(tmp_path: Path) -> None:
    result, console = _run_script(
        tmp_path,
        """\
        import gpurelay as relay
        total = 0
        for i in relay.range(3):
            total += i
        raise ValueError("boom in user code")
        """,
    )
    assert result.state is JobState.FAILED_USER
    assert [s.outcome for s in result.segments] == ["failed_user"]
    assert 'job_script.py", line 5' in console
    assert "ValueError: boom in user code" in console
    assert "gpurelay" not in console.split("Traceback")[1].split("ValueError")[0].replace(
        "job_script", ""
    )


def test_sys_exit_nonzero_is_user_error(tmp_path: Path) -> None:
    result, _ = _run_script(tmp_path, "import sys\nsys.exit(3)\n")
    assert result.state is JobState.FAILED_USER


def test_divergent_control_flow_is_non_resumable(tmp_path: Path) -> None:
    result, _console = _run_script(
        tmp_path,
        """\
        import os, time
        import gpurelay as relay
        if True:  # one top-level statement: the divergence is inside the resumed statement
            resumed = os.path.exists("flag.txt")  # restored working file changes control flow
            open("flag.txt", "w").write("x")
            if resumed:
                for _ in relay.range(1):
                    pass
            for i in relay.range(200):
                time.sleep(0.05)
        """,
        deadline_s=2.0,
    )
    assert result.state is JobState.FAILED_USER
    assert [s.outcome for s in result.segments] == ["handoff", "non_resumable"]
    assert "diverged" in result.message


def test_already_completed_job_is_not_rerun(tmp_path: Path) -> None:
    result, _ = _run_script(tmp_path, "print('hello')\n")
    assert result.state is JobState.COMPLETED
    assert (tmp_path / "final" / "log.txt").read_text() == "hello\n"
