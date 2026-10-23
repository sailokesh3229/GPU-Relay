"""Equivalence harness (spec §18): uninterrupted vs interrupted runs of the same job.

A run goes through the real runtime in subprocesses (Simulated backend), so hard kills
are real process kills. Outputs compared: merged log text, figures, and output files
(``*.pt`` compared tensor-by-tensor with ``torch.equal``, everything else byte-exact).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from gpurelay.backends.simulated import KillEvent, SimulatedBackend
from gpurelay.config import RuntimeSettings
from gpurelay.controller.local_runner import RunResult, run_local_job
from gpurelay.models import JobState
from gpurelay.storage import LocalFSStorage

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@dataclass
class RunOutput:
    result: RunResult
    final_dir: Path
    kills: list[KillEvent] = field(default_factory=list)

    @property
    def log(self) -> str:
        return (self.final_dir / "log.txt").read_text(encoding="utf-8")

    def files(self) -> dict[str, bytes]:
        root = self.final_dir / "files"
        return {
            p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }

    def figures(self) -> list[bytes]:
        return [p.read_bytes() for p in sorted((self.final_dir / "figures").glob("*.png"))]

    @property
    def outcomes(self) -> list[str]:
        return [s.outcome for s in self.result.segments]


def run_job(
    script: Path,
    root: Path,
    *,
    env: dict[str, str] | None = None,
    settings: RuntimeSettings | None = None,
    kill_after_s: float | None = None,
    kill_between_s: tuple[float, float] | None = None,
    deadline_s: float | None = None,
    seed: int = 0,
) -> RunOutput:
    """Run ``script`` to completion through the relay (uninterrupted if no faults given)."""
    storage = LocalFSStorage(root / "store")
    backend = SimulatedBackend(
        storage,
        root / "sessions",
        settings=settings or RuntimeSettings(),
        max_session_s=deadline_s or 7 * 24 * 3600,
        kill_after_s=kill_after_s,
        kill_between_s=kill_between_s,
        seed=seed,
        env=env,
    )
    try:
        result = run_local_job(script, storage=storage, backend=backend, out_dir=root / "final")
    finally:
        backend.cancel_timers()
    assert result.state is JobState.COMPLETED, (result.state, result.message, result.segments)
    return RunOutput(result=result, final_dir=root / "final", kills=list(backend.kills))


def _torch_equal(a: bytes, b: bytes) -> bool:
    import torch

    sa = torch.load(io.BytesIO(a), weights_only=True)
    sb = torch.load(io.BytesIO(b), weights_only=True)
    if isinstance(sa, dict):
        return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    return bool(torch.equal(sa, sb))


def notebook_outputs(run: RunOutput) -> list[list[dict]] | None:
    """Per-code-cell outputs of ``output.ipynb`` (execution counts removed)."""
    path = run.final_dir / "output.ipynb"
    if not path.exists():
        return None
    import json

    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        outs = []
        for out in cell["outputs"]:
            out = {k: v for k, v in out.items() if k != "execution_count"}
            # Object reprs embed memory addresses that differ between any two runs.
            if "data" in out and "text/plain" in out["data"]:
                text = "".join(out["data"]["text/plain"])
                out["data"] = {**out["data"], "text/plain": _ADDRESS.sub("0x?", text)}
            outs.append(out)
        cells.append(outs)
    return cells


_ADDRESS = __import__("re").compile(r"0x[0-9a-fA-F]{6,}")


def assert_equivalent(expected: RunOutput, actual: RunOutput) -> None:
    """Identical merged log, figures, output files, and notebook outputs."""
    assert notebook_outputs(actual) == notebook_outputs(expected)
    assert actual.log == expected.log
    assert actual.figures() == expected.figures()
    files_a, files_b = expected.files(), actual.files()
    assert files_a.keys() == files_b.keys()
    for name in files_a:
        if name.endswith(".pt"):
            assert _torch_equal(files_a[name], files_b[name]), name
        else:
            assert files_a[name] == files_b[name], name


def lost_work_s(run: RunOutput) -> list[float]:
    """Seconds between the last committed snapshot and each kill that followed it
    within the same segment (kills before a segment's first commit are excluded)."""
    lost = []
    for kill in run.kills:
        if kill.last_commit_created_at is not None and kill.last_commit_segment == kill.segment_id:
            lost.append(kill.killed_at - kill.last_commit_created_at.timestamp())
    return lost
