"""Execution positions, resumable loops, and replay-to-resume-point.

Model
-----
A script is a sequence of top-level statements (see ``runtime/instrument.py``). Inside a
statement, execution is a tree of *contexts*: the statement itself and each iteration of a
relay loop. In a context, **events** (relay loop starts, explicit ``safe_point()`` calls,
framework ``fit``/``train`` calls) are numbered in order of occurrence. A position is::

    {"stmt": s, "end": "stmt"}                                  # statement s completed
    {"stmt": s, "path": [[e0, i0, id0], ...], "end": "iter"}    # iteration i_k of loop e_k done
    {"stmt": s, "path": [...], "end": "sp", "sp": e}            # explicit safe point e
    {"stmt": s, "path": [..., [e, 0, kind]], "end": "fw"}       # inside framework call e

While the target statement re-executes in **replay** mode:

* a loop whose event precedes the target's event in the same context is skipped;
* the target loop fast-forwards: for ``"iter"`` targets it restores state and continues
  live from iteration ``i+1``; for deeper targets it replays iteration ``i``;
* reaching a target safe point or framework call restores state and goes **live**.

In **passthrough** mode (statements re-executed as setup before the resume point) loops
run completely with no events or safe points.

**Adaptive loops** (Phase 9, hardening): an instrumented loop is tracked for its first
``PROBE_ITERATIONS`` iterations; if those take less than ``MIN_ITERATION_S`` each on average,
the rest of that loop instance runs with **sparse** safe points: iterations are not tracked
one by one (tracking costs ~2 us each), but every ``stride`` iterations (about
``SPARSE_INTERVAL_S``) the end of the current iteration is offered as a safe point. Loops
inside run untracked (no events), so positions stay deterministic.

Divergence from the checkpointed run raises :class:`ResumeMismatchError` (exit 88).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

PROBE_ITERATIONS = 32
MIN_ITERATION_S = 0.0005  # below this, per-iteration tracking (~2 us) is not worth it
SPARSE_INTERVAL_S = 0.05  # sparse mode: a safe point about this often

Position = dict[str, Any]
_Step = tuple[int, int, "str | None"]  # (event, iteration index, loop id / framework kind)


class ResumeMismatchError(BaseException):
    """Re-execution diverged from the run that produced the checkpoint (exit 88).

    A ``BaseException`` so that ``except Exception`` in user code cannot swallow it.
    """


@dataclass
class _Context:
    address: tuple[_Step, ...]
    events: int = 0


@dataclass
class _ActiveLoop:
    key: str
    event: int
    total: int | None
    completed: int = 0
    name: str | None = None


@dataclass
class LoopPlan:
    """How a loop instance should run."""

    event: int
    key: str
    loop_id: str | None = None
    skip: bool = False
    passthrough: bool = False
    start: int = 0  # first iteration index to yield
    restore_before_start: bool = False


@dataclass
class FrameworkPlan:
    """How a framework training call should run."""

    event: int
    kind: str
    skip: bool = False
    passthrough: bool = False
    resume: bool = False  # restore state, then resume from the saved framework checkpoint


def _key(address: tuple[_Step, ...], event: int) -> str:
    parts = [f"L{e}[{i}]" for e, i, _ in address]
    parts.append(f"L{event}")
    return "/".join(parts)


def _parse_path(raw: list[Any]) -> list[_Step]:
    steps: list[_Step] = []
    for entry in raw:
        event, index = int(entry[0]), int(entry[1])
        ident = entry[2] if len(entry) > 2 else None
        steps.append((event, index, ident))
    return steps


@dataclass
class Tracker:
    """Tracks the current position and drives replay toward ``target``."""

    on_reach: Callable[[], None]
    target: Position | None = None
    stmt: int = 0
    passthrough: bool = False
    contexts: list[_Context] = field(default_factory=lambda: [_Context(())])
    active: list[_ActiveLoop] = field(default_factory=list)
    untracked: int = 0  # > 0 inside an adaptive loop that stopped tracking iterations

    # ------------------------------------------------------------------ statements

    def begin_statement(
        self, index: int, *, target: Position | None = None, passthrough: bool = False
    ) -> None:
        """Start executing top-level statement ``index`` (live, replay-to-target, or setup)."""
        self.stmt = index
        self.contexts = [_Context(())]
        self.active.clear()
        self.untracked = 0
        self.passthrough = passthrough
        self.target = target
        if target is not None:
            self._tpath = _parse_path(target.get("path", []))
            self._tend: str = target["end"]
            self._tsp: int | None = target.get("sp")
            if self._tend not in ("iter", "sp", "fw") or (
                self._tend in ("iter", "fw") and not self._tpath
            ):
                raise ResumeMismatchError(f"invalid resume position {target!r}")

    def end_statement(self) -> None:
        if self.target is not None:
            raise self._mismatch(f"statement {self.stmt} ended before the resume point")
        self.passthrough = False

    # ------------------------------------------------------------------ state

    @property
    def replaying(self) -> bool:
        return self.target is not None

    @property
    def current(self) -> _Context:
        return self.contexts[-1]

    def _reach(self) -> None:
        self.target = None
        self.on_reach()

    def _mismatch(self, what: str) -> ResumeMismatchError:
        return ResumeMismatchError(
            f"re-execution diverged from the checkpointed run ({what}); "
            "control flow before the resume point must be deterministic"
        )

    def _position(self, path: list[list[Any]], end: str, **extra: Any) -> Position:
        return {"stmt": self.stmt, "path": path, "end": end, **extra}

    def _match_event(self, event: int, ident: str | None, what: str) -> tuple[str, int]:
        """Compare an event with the target. Returns ("skip"|"match", target index)."""
        depth = len(self.current.address)
        if depth >= len(self._tpath):
            assert self._tsp is not None
            if event < self._tsp:
                return "skip", 0
            raise self._mismatch(f"{what} found where safe point {self._tsp} was expected")
        t_event, t_index, t_ident = self._tpath[depth]
        if event < t_event:
            return "skip", 0
        if event > t_event:
            raise self._mismatch(f"passed resume event {t_event} without entering it")
        if t_ident is not None and ident is not None and t_ident != ident:
            raise self._mismatch(f"{what} is {ident}, checkpoint expected {t_ident}")
        return "match", t_index

    # ------------------------------------------------------------------ loops

    def next_loop_key(self) -> str:
        """Key the next loop started in the current context will get."""
        return f"S{self.stmt}/" + _key(self.current.address, self.current.events)

    @property
    def nested(self) -> bool:
        """Inside an iteration of another tracked loop of this statement?"""
        return len(self.current.address) > 0

    def loop_start(self, total: int | None, loop_id: str | None = None) -> LoopPlan:
        if self.passthrough or self.untracked:
            return LoopPlan(event=-1, key="", passthrough=True)
        ctx = self.current
        event = ctx.events
        ctx.events += 1
        plan = LoopPlan(event=event, key=_key(ctx.address, event), loop_id=loop_id)
        if self.target is None:
            return plan
        verdict, t_index = self._match_event(event, loop_id, f"loop {loop_id or plan.key}")
        if verdict == "skip":
            plan.skip = True
            return plan
        depth = len(ctx.address)
        if depth == len(self._tpath) - 1 and self._tend == "iter":
            plan.start = t_index + 1
            plan.restore_before_start = True
        elif depth == len(self._tpath) - 1 and self._tend == "fw":
            raise self._mismatch(f"loop {plan.key} found where a framework call was expected")
        else:
            plan.start = t_index
        if total is not None and plan.start > total:
            raise self._mismatch(f"loop {plan.key} has {total} iterations, resume at {plan.start}")
        return plan

    def enter_loop(self, plan: LoopPlan, total: int | None, name: str | None) -> None:
        self.active.append(_ActiveLoop(plan.key, plan.event, total, plan.start, name))
        if plan.restore_before_start:
            self._reach()

    def exit_loop(self, plan: LoopPlan) -> None:
        if self.active and self.active[-1].key == plan.key:
            self.active.pop()

    def iteration_start(self, plan: LoopPlan, index: int) -> _Context:
        ctx = _Context((*self.current.address, (plan.event, index, plan.loop_id)))
        self.contexts.append(ctx)
        return ctx

    def iteration_end(self, plan: LoopPlan, index: int, ctx: _Context) -> Position | None:
        """Pop the iteration context; return the safe-point position if live."""
        self._pop(ctx)
        if self.active and self.active[-1].key == plan.key:
            self.active[-1].completed = index + 1
        if self.target is not None:
            raise self._mismatch(f"iteration {index} of {plan.key} ended before the resume point")
        return self._position([list(p) for p in ctx.address], "iter")

    def sparse_iteration_end(self, plan: LoopPlan, index: int) -> Position | None:
        """Sparse mode: iteration ``index`` (untracked, no context of its own) just ended."""
        if self.active and self.active[-1].key == plan.key:
            self.active[-1].completed = index + 1
        if self.target is not None:
            return None
        path = [*(list(step) for step in self.current.address), [plan.event, index, plan.loop_id]]
        return self._position(path, "iter")

    def abandon_iteration(self, ctx: _Context) -> None:
        """Iteration left via ``break``/exception: pop without a safe point."""
        self._pop(ctx)

    def _pop(self, ctx: _Context) -> None:
        if ctx in self.contexts[1:]:
            while self.contexts[-1] is not ctx:
                self.contexts.pop()
            self.contexts.pop()

    # ------------------------------------------------------------------ safe points

    def safe_point(self) -> Position | None:
        """Explicit safe point. Returns its position if live, None otherwise."""
        if self.passthrough or self.untracked:
            return None
        ctx = self.current
        event = ctx.events
        ctx.events += 1
        if self.target is None:
            return self._position([list(p) for p in ctx.address], "sp", sp=event)
        depth = len(ctx.address)
        if depth < len(self._tpath):
            if event < self._tpath[depth][0]:
                return None
            raise self._mismatch("passed the resume loop at an explicit safe point")
        assert self._tsp is not None
        if event < self._tsp:
            return None
        if event == self._tsp:
            self._reach()
            return None
        raise self._mismatch(f"passed resume safe point {self._tsp}")

    # ------------------------------------------------------------------ frameworks

    def framework_start(self, kind: str) -> FrameworkPlan:
        if self.passthrough or self.untracked:
            return FrameworkPlan(event=-1, kind=kind, passthrough=True)
        ctx = self.current
        event = ctx.events
        ctx.events += 1
        plan = FrameworkPlan(event=event, kind=kind)
        if self.target is None:
            return plan
        verdict, _ = self._match_event(event, kind, f"{kind} training call")
        if verdict == "skip":
            plan.skip = True
            return plan
        if len(ctx.address) != len(self._tpath) - 1 or self._tend != "fw":
            raise self._mismatch(f"{kind} training call found where a loop was expected")
        plan.resume = True
        self._reach()
        return plan

    def framework_position(self, plan: FrameworkPlan) -> Position:
        path = [list(p) for p in self.current.address] + [[plan.event, 0, plan.kind]]
        return self._position(path, "fw")

    # ------------------------------------------------------------------ progress

    def progress(self) -> dict[str, tuple[int, int | None]]:
        return {loop.name or loop.key: (loop.completed, loop.total) for loop in self.active}


def run_loop(
    tracker: Tracker,
    total: int | None,
    name: str | None,
    items_from: Callable[[int], Iterator[tuple[int, Any]]],
    on_safe_point: Callable[[Position], None],
    loop_id: str | None = None,
    *,
    adaptive: bool = False,
) -> Iterator[Any]:
    """Generator body shared by every resumable loop.

    ``items_from(start)`` must perform any fast-forward eagerly (before returning) and
    return an iterator over ``(index, item)`` beginning at ``start``. Fast-forwarding
    happens *before* state is restored, so RNG consumed while skipping is overwritten.
    ``adaptive`` loops stop tracking when their iterations are tiny (see module doc).
    """
    plan = tracker.loop_start(total, loop_id)
    if plan.passthrough:
        for _index, item in items_from(0):
            yield item
        return
    if plan.skip:
        return
    items = items_from(plan.start)
    tracker.enter_loop(plan, total, name)
    timed, began = 0, time.perf_counter()
    try:
        for index, item in items:
            ctx = tracker.iteration_start(plan, index)
            try:
                yield item
            except BaseException:
                tracker.abandon_iteration(ctx)
                raise
            position = tracker.iteration_end(plan, index, ctx)
            if position is not None:
                on_safe_point(position)
            if not adaptive or tracker.target is not None:
                continue
            timed += 1
            if timed == PROBE_ITERATIONS:
                elapsed = time.perf_counter() - began
                if elapsed >= PROBE_ITERATIONS * MIN_ITERATION_S:
                    adaptive = False  # iterations are long enough: keep tracking
                    continue
                stride = max(1, int(SPARSE_INTERVAL_S * PROBE_ITERATIONS / max(elapsed, 1e-9)))
                yield from _sparse(tracker, plan, items, stride, on_safe_point)
                break
    finally:
        tracker.exit_loop(plan)


def _sparse(
    tracker: Tracker,
    plan: LoopPlan,
    items: Iterator[tuple[int, Any]],
    stride: int,
    on_safe_point: Callable[[Position], None],
) -> Iterator[Any]:
    """The rest of a loop with tiny iterations: a safe point every ``stride`` iterations."""
    tracker.untracked += 1
    try:
        countdown = stride
        for index, item in items:
            yield item
            countdown -= 1
            if countdown:
                continue
            countdown = stride
            tracker.untracked -= 1
            try:
                position = tracker.sparse_iteration_end(plan, index)
                if position is not None:
                    on_safe_point(position)
            finally:
                tracker.untracked += 1
    finally:
        tracker.untracked -= 1
