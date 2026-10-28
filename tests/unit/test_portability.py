from __future__ import annotations

import ast
import json
from pathlib import Path

from gpurelay.analyzer import analyze_path
from gpurelay.analyzer.portability import will_fail


def _errors(*cells: str) -> list[tuple[str, int | None, int | None]]:
    trees = [ast.parse(c) for c in cells]
    return [(e.kind, e.cell, e.line) for e in will_fail(trees, cells=len(cells) > 1)]


def test_colab_only_code_is_reported() -> None:
    found = _errors("from google.colab import drive\ndrive.mount('/content/drive')\n")
    assert ("colab_only", None, 1) in found and ("colab_path", None, 2) in found
    assert ("colab_only", None, 1) in _errors("import google.colab.files as f\n")


def test_names_used_but_never_defined() -> None:
    found = _errors("x = 1\n", "for batch in train_loader:\n    print(batch, x)\n")
    assert found == [("undefined_name", 2, 1)]


def test_valid_code_reports_nothing() -> None:
    code = """
import os, numpy as np
from math import *  # noqa
"""
    assert _errors(code + "print(pi)\n") == []  # a star import: names cannot be known
    valid = """
import os
def f(a, *args, b=1, **kw):
    global G
    G = [y for y in range(a)]
    lam = lambda q: q + b
    try:
        pass
    except ValueError as err:
        print(err)
    with open(__file__) as fh:
        return fh, lam, args, kw, os
class C:
    pass
match f(1):
    case {"k": v, **rest}:
        print(v, rest)
    case [first, *others]:
        print(first, others)
    case C() as c:
        print(c)
if (n := 3) > 2:
    print(n, G, display, get_ipython)
"""
    assert _errors(valid) == []


def test_the_infovae_notebook_is_flagged_before_submit(tmp_path: Path) -> None:
    cells = [
        "from google.colab import drive\ndrive.mount('/content/drive')\n!pip install pytorch-fid",
        "import os\nBASE_DIR = '/content/drive/MyDrive/InfoVAE'",
        "for epoch in range(3):\n    for data, _ in train_loader:\n        pass\n"
        "    fid = compute(val_loader)",
    ]
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
          "cells": [{"cell_type": "code", "metadata": {}, "source": c, "outputs": [],
                     "execution_count": None} for c in cells]}  # fmt: skip
    path = tmp_path / "infovae.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    errors = analyze_path(path).errors
    kinds = {(e.kind, e.cell) for e in errors}
    assert ("colab_only", 1) in kinds and ("colab_path", 2) in kinds
    names = {e.message.split("`")[1] for e in errors if e.kind == "undefined_name"}
    assert names == {"train_loader", "val_loader", "compute"}
