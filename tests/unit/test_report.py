"""The ``gpurelay analyze`` report: text from the job is shown as written (rich would read
``[...]`` as markup and drop it)."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from gpurelay.analyzer import analyze_path
from gpurelay.analyzer.report import render


def _text(path: Path) -> str:
    out = io.StringIO()
    render(analyze_path(path), Console(file=out, width=200, color_system=None))
    return out.getvalue()


def test_brackets_in_code_and_hints_are_kept(tmp_path: Path) -> None:
    script = tmp_path / "[draft] train.py"
    script.write_text(
        "batches = {'key': [1, 2, 3]}\n"
        "total = 0\n"
        "for x in batches['key']:\n"
        "    total += x\n"
        "def train():\n"
        "    for epoch in range(3):\n"
        "        pass\n"
        "train()\n",
        encoding="utf-8",
    )
    text = _text(script)
    assert "[draft] train.py" in text
    assert "for x in batches['key']" in text
    assert "under [runtime]" in text  # the instrument_functions hint names its section
