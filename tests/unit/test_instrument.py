"""Layer C instrumentation: which loops are rewritten, what is recorded, line numbers."""

from __future__ import annotations

import traceback
from collections.abc import Iterator
from typing import Any

import pytest

from gpurelay.runtime.instrument import (
    RELAY_NAME,
    InstrumentOptions,
    called_functions,
    plan_script,
)

SOURCE = '''\
"""module doc"""
from __future__ import annotations
import os
import gpurelay as relay
from gpurelay import chunks as ch

losses = []
def helper(x):
    global counter
    while x:
        x -= 1
    return x

with open(os.devnull) as fh:
    for i, b in enumerate(range(3)):
        losses.append(b)
        table[i] = b
        model.weight.add_(1)
for e in relay.range(2):
    pass
for c in ch([1, 2], size=1):
    pass
while False:
    pass
if __name__ == "__main__":
    for k in range(2):
        for inner in range(2):
            helper(losses)
os.remove("x")
'''


class FakeRelay:
    """Records instrumented loop calls and passes items through."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def loop(self, loop_id: str, iterable: Any) -> Iterator[Any]:
        self.calls.append(loop_id)
        return iter(iterable)


def _plan(
    source: str = SOURCE,
    filename: str = "/abs/one/job.py",
    options: InstrumentOptions | None = None,
) -> Any:
    return plan_script(source, filename, loop_file_id="job.py", options=options)


def test_outermost_for_loops_are_instrumented() -> None:
    plan = _plan(options=InstrumentOptions(loop_depth=1))  # v1 behaviour
    by_line = {s.lineno: s for s in plan.statements}
    assert [loop.lineno for loop in by_line[14].loops] == [15]  # inside `with`
    assert [loop.kind for loop in by_line[19].loops] == ["sdk"]  # relay.range: not wrapped
    assert [loop.kind for loop in by_line[21].loops] == ["sdk"]  # from-imported chunks
    assert [loop.kind for loop in by_line[23].uninstrumented] == ["while"]
    # Only the outermost loop of the `if __name__` block; never the nested one.
    assert [loop.lineno for loop in by_line[25].loops] == [26]
    assert plan.docstring == "module doc"
    assert plan.global_names == {"counter"}
    assert plan.sdk_aliases == {"relay"}


def test_nested_for_loops_are_instrumented_up_to_the_configured_depth() -> None:
    by_line = {s.lineno: s for s in _plan().statements}  # default: two levels
    assert [loop.lineno for loop in by_line[25].loops] == [26, 27]
    deep = (
        "for a in range(2):\n    for b in range(2):\n        for c in range(2):\n            pass\n"
    )
    assert [x.lineno for x in _plan(deep).statements[0].loops] == [1, 2]
    three = _plan(deep, options=InstrumentOptions(loop_depth=3)).statements[0].loops
    assert [x.lineno for x in three] == [1, 2, 3]
    # A loop nested in an SDK loop is instrumented too; one inside a while loop is not.
    sdk = (
        "import gpurelay as relay\nfor e in relay.range(2):\n    for b in range(3):\n        pass\n"
    )
    assert [x.kind for x in _plan(sdk).statements[1].loops] == ["sdk", "for"]
    loop_in_while = "while x:\n    for b in range(3):\n        pass\n"
    assert _plan(loop_in_while).statements[0].loops == []


def test_loop_ids_are_stable_across_working_directories() -> None:
    a = _plan(filename="/sessions/seg-1/work/job.py")
    b = _plan(filename="C:/elsewhere/seg-2/work/job.py")
    ids = lambda p: [loop.loop_id for s in p.statements for loop in s.loops]  # noqa: E731
    assert ids(a) == ids(b)
    assert all(i.startswith("L") for i in ids(a))


def test_assigned_and_mutated_names() -> None:
    plan = _plan()
    stmt = next(s for s in plan.statements if s.lineno == 14)
    assert {"fh", "i", "b", "losses", "table", "model"} <= stmt.assigned
    loop_stmt = next(s for s in plan.statements if s.lineno == 25)
    assert {"k", "inner"} <= loop_stmt.assigned
    assert "losses" in loop_stmt.call_args
    assert [f.name for f in called_functions(loop_stmt, plan)] == ["helper"]


def test_side_effects_and_definitions() -> None:
    plan = _plan()
    last = plan.statements[-1]
    assert last.side_effects and "file deletion" in last.side_effects[0]
    defs = [s.kind for s in plan.statements if s.is_definition]
    assert defs == ["ImportFrom", "Import", "Import", "ImportFrom", "FunctionDef"]


def test_instrumented_code_runs_and_calls_relay() -> None:
    source = "out = []\nfor x in range(3):\n    out.append(x * x)\n"
    plan = _plan(source)
    relay = FakeRelay()
    ns: dict[str, Any] = {RELAY_NAME: relay}
    for stmt in plan.statements:
        exec(stmt.code, ns)
    assert ns["out"] == [0, 1, 4]
    assert relay.calls == [plan.statements[1].loops[0].loop_id]


def test_future_annotations_apply_to_every_statement() -> None:
    source = "from __future__ import annotations\ndef f(x: Undefined) -> Missing:\n    return x\n"
    ns: dict[str, Any] = {}
    for stmt in _plan(source).statements:
        exec(stmt.code, ns)  # would raise NameError without the future flag
    assert ns["f"].__annotations__ == {"x": "Undefined", "return": "Missing"}


def test_traceback_line_numbers_match_the_original_file() -> None:
    source = (
        "values = []\n"
        "for i in range(5):\n"
        "    values.append(i)\n"
        "    if i == 3:\n"
        "        raise ValueError('deliberate')\n"
    )
    plan = plan_script(source, "/work/job.py", loop_file_id="job.py")
    ns: dict[str, Any] = {RELAY_NAME: FakeRelay()}
    with pytest.raises(ValueError) as info:
        for stmt in plan.statements:
            exec(stmt.code, ns)
    frames = traceback.extract_tb(info.value.__traceback__)
    user = [f for f in frames if f.filename == "/work/job.py"]
    assert [f.lineno for f in user] == [5]


def test_syntax_error_reports_location() -> None:
    with pytest.raises(SyntaxError) as info:
        plan_script("x = (\n", "/work/job.py")
    assert info.value.filename == "/work/job.py"
