"""Analyzer snapshot tests (plan Phase 3): expected ``analysis.json`` per corpus item.

Regenerate after an intended change with ``GPURELAY_UPDATE_SNAPSHOTS=1 pytest``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpurelay.analyzer import analyze_path, analyze_script
from gpurelay.models import ResumabilityLevel
from gpurelay.runtime.instrument import InstrumentOptions

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
EXPECTED = CORPUS / "expected"
ITEMS = sorted(p.name for p in [*CORPUS.glob("*.py"), *CORPUS.glob("*.ipynb")])


def _stable(data: dict[str, Any]) -> dict[str, Any]:
    """Drop machine-dependent fields (installed library versions)."""
    data = dict(data)
    data["frameworks"] = {
        fw: {k: v for k, v in info.items() if k == "supported_majors"}
        for fw, info in data["frameworks"].items()
    }
    return data


@pytest.mark.parametrize("name", ITEMS)
def test_analysis_snapshot(name: str) -> None:
    actual = _stable(analyze_path(CORPUS / name).to_json())
    snapshot = EXPECTED / f"{Path(name).stem}.analysis.json"
    if os.environ.get("GPURELAY_UPDATE_SNAPSHOTS") or not snapshot.exists():
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    assert actual == json.loads(snapshot.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("name", "level"),
    [
        ("mlp", "FULL"),
        ("vae_fid", "FULL"),
        ("hf_trainer", "FULL"),
        ("lightning_tiny", "FULL"),
        ("keras_tiny", "FULL"),
        ("chunked_eval", "FULL"),
        ("sdk_mlp", "FULL"),
        ("sdk_chunks", "FULL"),
        ("while_loop", "PARTIAL"),
        ("mlp_notebook.ipynb", "FULL"),
        ("vae_fid_notebook.ipynb", "FULL"),
        ("chunked_eval_notebook.ipynb", "FULL"),
    ],
)
def test_corpus_levels(name: str, level: str) -> None:
    path = CORPUS / (name if name.endswith(".ipynb") else f"{name}.py")
    assert analyze_path(path).overall.value == level


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "job.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_main_function_pattern_is_restart_only(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "def main():\n    for i in range(10):\n        pass\n\n"
        "if __name__ == '__main__':\n    main()\n",
    )
    analysis = analyze_script(path)
    assert analysis.overall is ResumabilityLevel.RESTART_ONLY
    assert "instrument_functions = true" in analysis.blocks[0].notes[0]
    opted_in = analyze_script(path, options=InstrumentOptions(functions=True))
    assert opted_in.overall is ResumabilityLevel.FULL
    assert opted_in.blocks[0].covered_by == ["C (loops in functions)"]


def test_while_loops_with_opt_in_are_full() -> None:
    assert analyze_script(CORPUS / "while_loop.py").overall is ResumabilityLevel.PARTIAL
    opted_in = analyze_script(CORPUS / "while_loop.py", options=InstrumentOptions(while_loops=True))
    assert opted_in.overall is ResumabilityLevel.FULL
    assert [b["kind"] for blk in opted_in.blocks for b in blk.loops] == ["for", "while"]


def test_framework_call_counts_as_covered(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "import lightning.pytorch as L\ntrainer = L.Trainer()\ntrainer.fit(object())\n",
    )
    analysis = analyze_script(path)
    assert analysis.overall is ResumabilityLevel.FULL
    assert analysis.blocks[0].covered_by == ["B (lightning hook)"]
    assert "lightning" in analysis.frameworks


def test_risks_and_suggestions(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "import os, random, torch\n"
        "from torch.utils.data import DataLoader\n"
        "torch.backends.cudnn.benchmark = True\n"
        "it = iter([1, 2])\n"
        "x = random.random()\n"
        "loader = DataLoader([1], num_workers=0)\n"
        "os.remove('tmp.txt')\n"
        "loss = torch.zeros(1, requires_grad=True).sum()\n"
        "loss.backward()\n",
    )
    analysis = analyze_script(path)
    kinds = {r.kind for r in analysis.risks}
    assert {"nondeterminism", "unpicklable_state", "side_effect"} <= kinds
    tips = {s.kind for s in analysis.suggestions}
    assert {"mixed_precision", "dataloader_workers", "pin_memory"} <= tips


def test_missing_requirements(tmp_path: Path) -> None:
    path = _write(tmp_path, "import numpy\nimport sklearn\nimport json\nimport yaml\n")
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("numpy>=1.24\nPyYAML==6.0  # config\n", encoding="utf-8")
    env = analyze_script(path, requirements=reqs).environment
    assert env["missing_from_requirements"] == ["sklearn"]


def test_notebook_cells_shell_commands_and_setup(tmp_path: Path) -> None:
    import nbformat
    from nbformat.v4 import new_code_cell, new_notebook

    nb = new_notebook(
        cells=[
            new_code_cell("%pip install tqdm\nimport os"),
            new_code_cell("!wget -q https://example.com/data.zip"),
            new_code_cell("total = 0\nwhile total < 3:\n    total += 1"),
        ]
    )
    path = tmp_path / "nb.ipynb"
    nbformat.write(nb, str(path))
    analysis = analyze_path(path)
    assert analysis.kind == "notebook"
    by_cell = {b.cell: b for b in analysis.blocks}
    assert "installs packages" in by_cell[1].notes[0]
    assert "shell command" in by_cell[2].notes[0]
    assert by_cell[3].level is ResumabilityLevel.PARTIAL
    assert any(r.kind == "side_effect" and r.cell == 2 for r in analysis.risks)
