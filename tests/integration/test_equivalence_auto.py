"""Phase 3 verification: UNMODIFIED corpus items survive kills/handoffs with identical outputs.

Sessions on this test machine need several seconds before user code runs (interpreter,
torch, framework imports), so torch/framework scenarios use sessions of 10-25 s.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from equivalence import CORPUS, RunOutput, assert_equivalent, lost_work_s, run_job

from gpurelay.analyzer import analyze_script
from gpurelay.backends.simulated import SimulatedBackend
from gpurelay.config import RuntimeSettings
from gpurelay.controller.local_runner import run_local_job
from gpurelay.models import JobState
from gpurelay.storage import LocalFSStorage

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]


def _settings(margin: float = 0.3, **extra: Any) -> RuntimeSettings:
    # A low replay threshold exercises statement skipping on resume.
    return RuntimeSettings(
        min_interval_s=0.3,
        max_lost_work_s=1.0,
        safety_margin_s=margin,
        heartbeat_s=1,
        replay_threshold_s=0.5,
        **extra,
    )


def _pair(
    tmp_path: Path,
    name: str,
    env: dict[str, str],
    margin: float = 0.3,
    options: dict[str, Any] | None = None,
    **faults: object,
) -> tuple[RunOutput, RunOutput]:
    script = CORPUS / f"{name}.py"
    settings = _settings(margin, **(options or {}))
    baseline = run_job(script, tmp_path / "a", env=env, settings=settings)
    assert baseline.outcomes == ["completed"]
    run = run_job(script, tmp_path / "b", env=env, settings=settings, **faults)  # type: ignore[arg-type]
    return baseline, run


# ---------------------------------------------------------------- no ML framework


def test_chunked_eval_unmodified_random_kills(tmp_path: Path) -> None:
    env = {"CORPUS_ITEMS": "800", "CORPUS_CHUNK_SLEEP": "0.1"}
    baseline, run = _pair(tmp_path, "chunked_eval", env, kill_between_s=(1.2, 2.4), seed=1)
    assert run.outcomes.count("crashed") >= 2, run.outcomes
    assert_equivalent(baseline, run)
    assert max(lost_work_s(run), default=0) <= 1.0 + 0.1 + 1.0


def test_while_loop_block_restarts_correctly(tmp_path: Path) -> None:
    assert analyze_script(CORPUS / "while_loop.py").overall.value == "PARTIAL"
    env = {"CORPUS_STEP_SLEEP": "0.08"}
    baseline, run = _pair(tmp_path, "while_loop", env, kill_between_s=(3.2, 4.5), seed=4)
    assert "crashed" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)


def test_rng_objects_from_replayed_setup_resume_their_stream(tmp_path: Path) -> None:
    env = {"CORPUS_STEP_SLEEP": "0.2"}
    baseline, run = _pair(tmp_path, "rng_objects", env, kill_between_s=(2.0, 3.0), seed=9)
    assert run.outcomes.count("crashed") >= 2, run.outcomes
    assert_equivalent(baseline, run)


def test_imports_inside_try_and_if_blocks_survive_resume(tmp_path: Path) -> None:
    env = {"CORPUS_STEP_SLEEP": "0.3"}
    baseline, run = _pair(tmp_path, "conditional_imports", env, kill_between_s=(1.6, 2.6), seed=2)
    assert run.outcomes.count("crashed") >= 2, run.outcomes
    assert_equivalent(baseline, run)


def test_while_loop_opt_in_resumes_inside_the_loop(tmp_path: Path) -> None:
    env = {"CORPUS_STEP_SLEEP": "0.08"}
    options = {"instrument_while": True}
    baseline, run = _pair(tmp_path, "while_loop", env, options=options,
                          kill_between_s=(1.8, 2.6), seed=4)  # fmt: skip
    assert run.outcomes.count("crashed") >= 2, run.outcomes
    assert_equivalent(baseline, run)


def test_function_loops_opt_in_restore_locals(tmp_path: Path) -> None:
    env = {"CORPUS_STEP_SLEEP": "0.1"}
    options = {"instrument_functions": True, "instrument_while": True}
    baseline, run = _pair(tmp_path, "function_loops", env, options=options,
                          kill_between_s=(1.5, 2.5), seed=6)  # fmt: skip
    assert run.outcomes.count("crashed") >= 1, run.outcomes
    assert_equivalent(baseline, run)


def test_error_inside_instrumented_loop_keeps_line_numbers(tmp_path: Path) -> None:
    script = tmp_path / "job_script.py"
    script.write_text(
        textwrap.dedent(
            """\
            total = 0
            for i in range(10):
                total += i
                if i == 4:
                    raise RuntimeError("deliberate failure")
            """
        )
    )
    storage = LocalFSStorage(tmp_path / "store")
    backend = SimulatedBackend(storage, tmp_path / "sessions", settings=_settings())
    console: list[str] = []
    result = run_local_job(
        script,
        storage=storage,
        backend=backend,
        out_dir=tmp_path / "final",
        on_console=console.append,
    )
    text = "".join(console)
    assert result.state is JobState.FAILED_USER
    assert 'job_script.py", line 5, in <module>' in text
    assert "__relay__" not in text and "gpurelay" not in text.split("Traceback")[1].split(
        "RuntimeError"
    )[0].replace("job_script", "")


# ---------------------------------------------------------------- PyTorch


@pytest.mark.torch
def test_mlp_unmodified_kills_and_handoffs(tmp_path: Path) -> None:
    env = {"CORPUS_EPOCHS": "40", "CORPUS_EPOCH_SLEEP": "0.25"}
    baseline, run = _pair(tmp_path, "mlp", env, deadline_s=14, kill_between_s=(9.0, 16.0), seed=3)
    assert len(run.outcomes) >= 2, run.outcomes
    assert_equivalent(baseline, run)


@pytest.mark.torch
def test_torch_training_inside_a_function_opt_in(tmp_path: Path) -> None:
    # ~4 s of training; each session hands off 7 s after it starts (torch startup takes
    # several seconds), so the loop inside train() resumes at least once.
    env = {"CORPUS_STEP_SLEEP": "0.1"}
    options = {"instrument_functions": True}
    baseline, run = _pair(tmp_path, "function_loops_torch", env, options=options,
                          deadline_s=7.0)  # fmt: skip
    assert run.outcomes.count("handoff") >= 1, run.outcomes
    assert_equivalent(baseline, run)


@pytest.mark.torch
def test_dataloader_resumes_mid_epoch_with_the_same_batches(tmp_path: Path) -> None:
    # ~6.4 s per epoch; each session hands off 9 s after it starts (torch startup takes
    # several seconds of that), so it trains for well under an epoch: progress needs
    # mid-epoch resume, and equivalence needs the same shuffled batches after each resume.
    env = {"CORPUS_STEP_SLEEP": "0.05"}
    baseline, run = _pair(tmp_path, "dataloader_mid_epoch", env, deadline_s=9.0)
    assert run.outcomes.count("handoff") >= 3, run.outcomes
    assert_equivalent(baseline, run)


@pytest.mark.torch
def test_vae_fid_unmodified_survives_random_kills(tmp_path: Path) -> None:
    """Plan Phase 3 'done when': the owner's use case, zero code changes."""
    env = {"CORPUS_EPOCHS": "24", "CORPUS_EPOCH_SLEEP": "0.4"}
    baseline, run = _pair(tmp_path, "vae_fid", env, kill_between_s=(8.0, 10.5), seed=5)
    # The number of kills depends on machine speed; at least one is required.
    assert "crashed" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)


# ---------------------------------------------------------------- frameworks (Layer B)


@pytest.mark.hf
def test_hf_trainer_unmodified_handoffs(tmp_path: Path) -> None:
    env = {"CORPUS_EPOCHS": "4", "CORPUS_STEP_SLEEP": "0.2"}
    baseline, run = _pair(tmp_path, "hf_trainer", env, margin=2.0, deadline_s=22)
    assert "handoff" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)
    assert not (run.final_dir / "files" / "hf_out").exists()  # relay saves never leak


@pytest.mark.lightning
def test_lightning_unmodified_handoffs(tmp_path: Path) -> None:
    env = {"CORPUS_EPOCHS": "16", "CORPUS_EPOCH_SLEEP": "0.8"}
    baseline, run = _pair(tmp_path, "lightning_tiny", env, margin=1.5, deadline_s=18)
    assert "handoff" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)


@pytest.mark.keras
def test_keras_unmodified_handoffs(tmp_path: Path) -> None:
    # ~12 s of epochs plus Keras start-up: always more than one 16 s session, however fast
    # the imports are (with 16 epochs a warm start finished in one session, no handoff).
    env = {"CORPUS_EPOCHS": "24", "CORPUS_EPOCH_SLEEP": "0.5", "KERAS_BACKEND": "torch"}
    baseline, run = _pair(tmp_path, "keras_tiny", env, margin=1.5, deadline_s=16)
    assert "handoff" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)
