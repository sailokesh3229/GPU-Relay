"""Notebook planning and output-notebook assembly (Phase 4)."""

from __future__ import annotations

import ast
from typing import Any

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from gpurelay.runtime.executor_notebook import build_output_notebook
from gpurelay.runtime.instrument import RELAY_NAME, CellInstrumenter, cell_file_id, plan_cells

IPython = pytest.importorskip("IPython")
from IPython.core.inputtransformer2 import TransformerManager  # noqa: E402

transform = TransformerManager().transform_cell

CELLS = [
    "%env A=1\nimport os\nx = 1",  # mixed: imports + other statements
    "def f(v):\n    return v * 2",  # definitions only
    "%pip install tqdm",  # environment setup
    "!echo hi > out.txt",  # shell side effect
    "total = 0\nfor i in range(3):\n    total += f(i)\ntotal",  # instrumented loop
]


def test_plan_cells_classifies_cells() -> None:
    plan = plan_cells(CELLS, transform, name="nb.ipynb")
    mixed, defs, setup, shell, loop = plan.statements
    assert mixed.definitions_code is not None and not mixed.is_definition
    assert defs.is_definition and defs.definitions_code is None
    assert setup.is_setup and not setup.side_effects
    assert shell.side_effects and "shell command" in shell.side_effects[0]
    assert [lp.kind for lp in loop.loops] == ["for"]
    assert {"total", "i"} <= loop.assigned
    assert "f" in plan.functions


def test_definitions_code_recreates_imports_only() -> None:
    plan = plan_cells(CELLS, transform)
    ns: dict[str, Any] = {}
    code = plan.statements[0].definitions_code
    assert code is not None
    exec(code, ns)
    assert "os" in ns and "x" not in ns


def test_runtime_instrumenter_matches_plan_loop_ids() -> None:
    plan = plan_cells(CELLS, transform)
    instrumenter = CellInstrumenter(plan)
    instrumenter.cell = 4
    tree = instrumenter.visit(ast.parse(transform(CELLS[4])))
    calls = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == RELAY_NAME
    ]
    assert calls == [plan.statements[4].loops[0].loop_id]
    assert cell_file_id(4) in "cell-4"


def test_build_output_notebook() -> None:
    source = new_notebook(
        cells=[new_markdown_cell("# t"), new_code_cell("print(1)"), new_code_cell("2"),
               new_code_cell("never run")]
    )  # fmt: skip
    outputs = {
        0: [{"output_type": "stream", "name": "stdout", "text": "1\n"}],
        1: [
            {
                "output_type": "execute_result",
                "data": {"text/plain": "2"},
                "metadata": {},
                "execution_count": None,
            }
        ],
    }
    out = build_output_notebook(source, outputs)
    nbformat.validate(out)
    code = [c for c in out.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == [1, 2, None]
    assert code[1].outputs[0].execution_count == 2
    assert code[2].outputs == []
    assert source.cells[1].outputs == []  # source untouched
