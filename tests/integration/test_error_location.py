"""Failed jobs say where in the user's code they failed (the job page points at that line)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
import pytest

from gpurelay.backends.simulated import SimulatedBackend
from gpurelay.config import RuntimeSettings
from gpurelay.controller.local_runner import run_local_job
from gpurelay.models import JobState
from gpurelay.runtime.errorloc import parse_location
from gpurelay.storage import LocalFSStorage

pytestmark = pytest.mark.integration

FAST = RuntimeSettings(min_interval_s=0.2, max_lost_work_s=0.5, safety_margin_s=0.3, heartbeat_s=1)


def _run(tmp_path: Path, job: Path) -> object:
    storage = LocalFSStorage(tmp_path / "store")
    backend = SimulatedBackend(storage, tmp_path / "sessions", settings=FAST, max_session_s=3600)
    return run_local_job(job, storage=storage, backend=backend, out_dir=tmp_path / "final")


def test_a_script_error_names_the_failing_line_inside_a_function(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        textwrap.dedent("""\
        import math

        def ratio(a, b):
            return a / b

        for i in range(3):
            print("step", i)
            print(ratio(1, i - 2))
        """)
    )
    result = _run(tmp_path, script)
    assert result.state is JobState.FAILED_USER  # type: ignore[attr-defined]
    assert result.message == "line 4: ZeroDivisionError: division by zero"  # type: ignore[attr-defined]
    assert parse_location(result.message) == (None, 4)  # type: ignore[attr-defined]


def test_a_notebook_error_names_the_cell_and_line_where_it_happened(tmp_path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import os\n!echo setup"),
        nbformat.v4.new_code_cell("def load():\n    x = 1\n    return train_loader + x"),
        nbformat.v4.new_markdown_cell("notes"),  # not counted: code cells only
        nbformat.v4.new_code_cell("for epoch in range(2):\n    print(epoch)\n    data = load()"),
    ]
    path = tmp_path / "job.ipynb"
    nbformat.write(nb, str(path))
    result = _run(tmp_path, path)
    assert result.state is JobState.FAILED_USER  # type: ignore[attr-defined]
    assert result.message == (  # type: ignore[attr-defined]
        "cell 2, line 3: NameError: name 'train_loader' is not defined"
    )
