"""Phase 7 'done when', locally: a job relays automated backend -> Colab (manual) -> completes.

The automated leg is a Simulated backend whose budget covers one short session (it hands
off at its deadline); after that only Colab is left, so the controller asks the user. The
test plays the user: it runs the generated notebook's own check cells (with fake Colab/Hub
modules reading the same storage), then the runtime the way the notebook's last cell does.
The relayed outputs must be equivalent to an uninterrupted run, and re-running the notebook
afterwards must refuse.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import nbformat
import pytest
from equivalence import CORPUS, RunOutput, assert_equivalent, run_job

from gpurelay.config import (
    BackendsSettings,
    ColabSettings,
    Config,
    LocalSettings,
    NotifySettings,
    RuntimeSettings,
    SchedulerSettings,
    SimulatedSettings,
    StorageSettings,
)
from gpurelay.controller.service import Context, open_context, submit_job
from gpurelay.models import JobState, SegmentState
from gpurelay.storage.factory import storage_uri

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

RUNTIME = RuntimeSettings(
    min_interval_s=0.2, max_lost_work_s=0.6, safety_margin_s=0.3, heartbeat_s=1,
    replay_threshold_s=0.5,
)  # fmt: skip
ENV = {"CORPUS_ITEMS": "800", "CORPUS_CHUNK_SLEEP": "0.1"}  # longer than one 3 s session
SCRIPT = "chunked_eval.py"
SECRET = "hf_test_token"


def _config(home: Path) -> Config:
    return Config(
        runtime=RUNTIME,
        scheduler=SchedulerSettings(min_segment_h=0.0004, poll_interval_s=1),
        storage=StorageSettings(kind="localfs", localfs_root=home / "store"),
        notify=NotifySettings(desktop=False),
        backends=BackendsSettings(
            local=LocalSettings(enabled=False),
            # 3.24 s of budget: one session (3 s deadline, hands off a bit earlier), then
            # less than min_segment_h (1.44 s) is left.
            simulated=[SimulatedSettings(name="sim", budget_hours=0.0009, max_session_s=3.0)],
            colab_manual=ColabSettings(enabled=True, max_session_s=3600),
        ),
    )


def _tick_until(ctx: Context, done: Any, timeout_s: float = 120) -> None:
    end = time.monotonic() + timeout_s
    while not done():
        if time.monotonic() > end:
            job = ctx.db.list_jobs()[0]
            segs = [
                (s.backend, s.state.value, s.outcome, s.message) for s in ctx.db.segments(job.id)
            ]
            raise AssertionError(f"timed out: {job.state.value} {job.message!r} {segs}")
        ctx.controller.tick()
        time.sleep(0.3)


def _notebook_checks(
    ctx: Context, notebook: Path, monkeypatch: pytest.MonkeyPatch, tmp: Path
) -> dict[str, Any]:
    """Run the notebook's code cells up to (not including) the runtime install."""

    def hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        data = ctx.storage.get_bytes(filename)
        if data is None:
            error = FileNotFoundError(filename)
            error.response = types.SimpleNamespace(status_code=404)  # type: ignore[attr-defined]
            raise error
        out = tmp / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: SECRET)  # type: ignore[attr-defined]
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cells = [c.source for c in nbformat.read(str(notebook), as_version=4).cells
             if c.cell_type == "code"]  # fmt: skip
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for source in cells[:2]:
        exec(compile(source, str(notebook), "exec"), namespace)
    return namespace


def test_relay_to_colab_and_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    home = tmp_path / "home"
    ctx = open_context(_config(home), home)
    colab = ctx.controller.backends["colab_manual"].backend
    colab.storage_uri = "hf:test/store"  # the notebook's Hub reads are faked onto local storage
    job = submit_job(ctx, CORPUS / SCRIPT)

    # 1. Automated leg: the Simulated backend hands off; then its budget is gone.
    _tick_until(ctx, lambda: ctx.db.get_job(job.id).state is JobState.AWAITING_USER)  # type: ignore[union-attr]
    assert ctx.db.segments(job.id)[0].outcome == "handoff"
    [waiting] = [s for s in ctx.db.segments(job.id) if s.state is SegmentState.AWAITING_USER]
    assert waiting.backend == "colab_manual" and waiting.start_checkpoint_seq is not None

    # 2. The user opens the notebook; its checks pick the requested segment.
    notebook = colab.notebook_path(job.id)
    ns = _notebook_checks(ctx, notebook, monkeypatch, tmp_path / "downloads")
    assert ns["SEGMENT_ID"] == waiting.id and ns["SEQ"] == waiting.start_checkpoint_seq

    # 3. ... and runs the job (what the last cell does, with local storage and paths).
    command = [
        sys.executable, "-m", "gpurelay.runtime.entry",
        "--job", job.id, "--segment", ns["SEGMENT_ID"],
        "--storage", storage_uri(ctx.storage),
        "--work-dir", str(tmp_path / "colab" / "job"),
        "--runtime-dir", str(tmp_path / "colab" / "runtime"),
        "--backend", "colab_manual",
        "--deadline", repr(time.time() + ns["DEADLINE_S"]),
        "--settings", json.dumps(ns["SETTINGS"]),
    ]  # fmt: skip
    env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg")
    proc = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _tick_until(ctx, lambda: proc.poll() is not None)
    assert proc.returncode == 0, proc.stderr.read().decode() if proc.stderr else ""
    _tick_until(ctx, lambda: ctx.db.get_job(job.id).state is JobState.COMPLETED)  # type: ignore[union-attr]

    segments = ctx.db.segments(job.id)
    assert [s.backend for s in segments][-1] == "colab_manual"
    assert segments[-1].outcome == "completed" and "sim" in {s.backend for s in segments}

    baseline = run_job(CORPUS / SCRIPT, tmp_path / "baseline", env=ENV, settings=RUNTIME)
    assert_equivalent(baseline, RunOutput(result=baseline.result,
                                          final_dir=home / "outputs" / job.id))  # fmt: skip

    # 4. Running the notebook again does nothing.
    with pytest.raises(Exception, match="already ran") as info:
        _notebook_checks(ctx, notebook, monkeypatch, tmp_path / "downloads2")
    assert type(info.value).__name__ == "GpuRelayStop"
