"""Opt-in instrumentation (Phase 9 item 3): ``while`` loops and loops inside functions."""

from __future__ import annotations

import itertools
import sys
import types
from collections.abc import Iterator
from typing import Any

from gpurelay.runtime.instrument import RELAY_NAME, InstrumentOptions, plan_script
from gpurelay.runtime.session import set_frame_locals

WHILE_ON = InstrumentOptions(while_loops=True)
FUNCS_ON = InstrumentOptions(functions=True)


class Passthrough:
    """``__relay__`` stand-in: records loops and passes items through."""

    def __init__(self) -> None:
        self.loops: list[tuple[str, tuple[str, ...] | None]] = []

    def loop(self, loop_id: str, iterable: Any, local_names: Any = None) -> Iterator[Any]:
        self.loops.append((loop_id, local_names))
        return iter(iterable)

    def forever(self) -> Iterator[int]:
        return itertools.count()


def _run(source: str, options: InstrumentOptions) -> tuple[dict[str, Any], Passthrough]:
    plan = plan_script(source, "job.py", options=options)
    relay = Passthrough()
    ns: dict[str, Any] = {RELAY_NAME: relay, "__name__": "__main__"}
    for stmt in plan.statements:
        assert stmt.code is not None
        exec(stmt.code, ns)
    return ns, relay


WHILE_PROGRAM = """
out = []
i = 0
while i < 10:
    i += 1
    if i == 3:
        continue
    if i == 8:
        break
    out.append(i)
j = 5
while j > 0:
    j -= 2
nested = []
k = 0
while k < 3:
    k += 1
    m = 0
    while m < k:
        m += 1
        nested.append((k, m))
"""


def test_instrumented_while_keeps_python_semantics() -> None:
    expected: dict[str, Any] = {}
    exec(WHILE_PROGRAM, expected)
    ns, relay = _run(WHILE_PROGRAM, WHILE_ON)
    for name in ("out", "i", "j", "nested", "k"):
        assert ns[name] == expected[name], name
    assert len(relay.loops) == 3 + 3  # three top-level whiles + the nested one, once per k
    plan = plan_script(WHILE_PROGRAM, "job.py", options=WHILE_ON)
    assert [loop.kind for s in plan.statements for loop in s.loops] == [
        "while", "while", "while", "while",
    ]  # fmt: skip


def test_while_with_else_and_default_options_are_not_instrumented() -> None:
    source = "n = 0\nwhile n < 3:\n    n += 1\nelse:\n    done = True\n"
    plan = plan_script(source, "job.py", options=WHILE_ON)
    assert [loop.kind for s in plan.statements for loop in s.uninstrumented] == ["while"]
    ns, relay = _run(source, WHILE_ON)
    assert ns["done"] is True and relay.loops == []
    default = plan_script(WHILE_PROGRAM, "job.py")
    assert not any(s.loops for s in default.statements)


FUNCTION_PROGRAM = """
counter = 0

def train(epochs, lr=0.1):
    global counter
    history = []
    w = 1.0
    for epoch in range(epochs):
        w -= lr * w
        history.append(w)
        counter += 1
    return history

def batches():
    for i in range(3):
        yield i

async def fetch():
    for i in range(3):
        pass

result = train(4)
items = list(batches())
"""


def test_function_loops_get_the_function_locals() -> None:
    ns, relay = _run(FUNCTION_PROGRAM, FUNCS_ON)
    assert ns["result"] == [0.9, 0.81, 0.7290000000000001, 0.6561000000000001]
    assert ns["items"] == [0, 1, 2] and ns["counter"] == 4
    [(_loop_id, names)] = relay.loops  # train() only: generators/async are not instrumented
    assert names == ("epoch", "history", "w")  # assigned locals; `counter` is global
    plan = plan_script(FUNCTION_PROGRAM, "job.py", options=FUNCS_ON)
    assert plan.functions["train"].instrumentable
    assert not plan.functions["batches"].instrumentable
    _ns, off = _run(FUNCTION_PROGRAM, InstrumentOptions())
    assert off.loops == []


def test_set_frame_locals_writes_into_a_running_function() -> None:
    seen: dict[str, Any] = {}

    def body() -> None:
        total = 1
        items = [1]
        frame = sys._getframe()
        set_frame_locals(frame, {"total": 42, "items": [7, 8]})
        seen.update(total=total, items=items)

    body()
    assert seen == {"total": 42, "items": [7, 8]}
    assert isinstance(sys._getframe(), types.FrameType)
