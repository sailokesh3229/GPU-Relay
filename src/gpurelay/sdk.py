"""Manual SDK (spec §7.4): explicit resumability for code the automatic layers miss.

    import gpurelay as relay

    state = relay.state(model=model, optimizer=opt, history=history)
    for epoch in relay.range(100):
        ...
    for chunk in relay.chunks(items, size=1000):
        relay.save_result(chunk.id, compute(chunk.items))
    results = relay.load_results()

Outside GPU Relay (plain ``python script.py``) every call is a cheap passthrough.

Contract inside GPU Relay: on resume the script re-runs from the top; finished relay
loops are skipped and the interrupted one fast-forwards. Anything needed after a
resume point (models, optimizers, accumulators such as ``history``) must be registered
with :func:`state`. Mutable containers (list/dict/set, arrays, tensors) are restored in
place; other values are read and written through the returned object
(``state.best_loss``).
"""

from __future__ import annotations

import builtins
import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpurelay.runtime.session import RelaySession

_session: RelaySession | None = None

# Passthrough mode (no session).
_local_state: dict[str, Any] = {}
_local_results: dict[str, dict[int, Any]] = {}
_local_active: list[str] = []
_local_last: list[str] = []
_local_counter = itertools.count()


def _activate(session: RelaySession | None) -> None:
    """Bind the SDK to a runtime session (called by the runtime, not by users)."""
    global _session
    _session = session


class SdkLoop:
    """Iterable returned by ``relay.range``/``relay.chunks`` under GPU Relay.

    Automatic instrumentation recognizes it and does not wrap the loop a second time.
    """

    def __init__(self, factory: Callable[[], Iterator[Any]]) -> None:
        self._factory = factory

    def __iter__(self) -> Iterator[Any]:
        return self._factory()


def is_active() -> bool:
    """True when running under GPU Relay."""
    return _session is not None


class State:
    """Attribute/item view over the registered state."""

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, Any]) -> None:
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self._values[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._values[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __repr__(self) -> str:
        return f"State({sorted(self._values)})"


def state(**objects: Any) -> State:
    """Register objects whose values must survive a resume; return the state view."""
    if _session is None:
        _local_state.update(objects)
        return State(_local_state)
    _session.register(objects)
    return State(_session.registry)


def range(*args: int, name: str | None = None) -> Iterable[int]:
    """Resumable ``range``: a safe point after every iteration."""
    r = builtins.range(*args)
    if _session is None:
        return r

    def items_from(start: int) -> Iterator[tuple[int, int]]:
        return ((i, r[i]) for i in builtins.range(start, len(r)))

    session = _session
    return SdkLoop(lambda: session.loop(len(r), name, items_from))


@dataclass(frozen=True)
class Chunk:
    """One chunk of work from :func:`chunks`."""

    id: int
    items: Sequence[Any]
    start: int  # index of the first item in the original sequence

    def __iter__(self) -> Iterator[Any]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def _chunk_source(items: Iterable[Any], size: int) -> tuple[int | None, Any]:
    if isinstance(items, Sequence):
        seq = items
        total = math.ceil(len(seq) / size)

        def from_sequence(start: int) -> Iterator[tuple[int, Chunk]]:
            for k in builtins.range(start, total):
                yield k, Chunk(k, seq[k * size : (k + 1) * size], k * size)

        return total, from_sequence

    def from_iterator(start: int) -> Iterator[tuple[int, Chunk]]:
        it = iter(items)
        # Fast-forward eagerly: consume (and discard) the items of completed chunks.
        for _ in itertools.islice(it, start * size):
            pass

        def batches() -> Iterator[tuple[int, Chunk]]:
            k = start
            while batch := list(itertools.islice(it, size)):
                yield k, Chunk(k, batch, k * size)
                k += 1

        return batches()

    return None, from_iterator


def chunks(
    items: Iterable[Any],
    size: int,
    *,
    name: str | None = None,
    parallel: bool = False,
    seed: int = 0,
) -> Iterable[Chunk]:
    """Resumable chunked iteration (evaluation, metrics, preprocessing).

    Pair with :func:`save_result` so finished chunks survive interruptions.

    ``parallel=True`` lets the controller compute chunks on several backends at once
    (results are merged in order). Then each chunk must be independent: it may only
    *return* data through :func:`save_result` (other variables it changes, files it writes,
    and figures it draws are not merged), and every chunk starts with all random number
    generators seeded from ``seed`` and its id, so results do not depend on where it runs.
    ``items`` must be a sequence (list, range, array, ...). The same seeding applies when
    the script runs without GPU Relay.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if parallel and not isinstance(items, Sequence):
        raise TypeError("relay.chunks(parallel=True) needs a sequence (list, range, array, ...)")
    total, source = _chunk_source(items, size)
    if parallel:
        assert total is not None

        def chunk_at(k: int) -> Chunk:
            chunk: Chunk = next(source(k))[1]
            return chunk

        if _session is None:
            return _local_parallel_chunks(total, chunk_at, name, seed)
        psession = _session
        return SdkLoop(lambda: psession.parallel_chunks(total, name, chunk_at, seed))
    if _session is None:
        return _local_chunks(source, name)
    session = _session
    return SdkLoop(lambda: session.loop(total, name, source, chunk_loop=True))


def _local_parallel_chunks(
    total: int, chunk_at: Callable[[int], Chunk], name: str | None, seed: int
) -> Iterator[Chunk]:
    """Outside GPU Relay: sequential, with the same per-chunk seeding as relayed runs."""
    from gpurelay.runtime.rng import chunk_seed, seed_all

    def seeded() -> Iterator[tuple[int, Chunk]]:
        for k in builtins.range(total):
            seed_all(chunk_seed(seed, k))
            yield k, chunk_at(k)

    yield from _local_chunks(lambda _start: seeded(), name)
    seed_all(chunk_seed(seed, total))


def _local_chunks(source: Any, name: str | None) -> Iterator[Chunk]:
    bucket = name or f"chunks-{next(_local_counter)}"
    _local_results[bucket] = {}
    _local_last[:] = [bucket]
    _local_active.append(bucket)
    try:
        for _index, chunk in source(0):
            yield chunk
    finally:
        _local_active.remove(bucket)


def save_result(chunk_id: int, value: Any) -> None:
    """Store the result of one chunk durably (survives cutoffs)."""
    if _session is not None:
        _session.save_result(chunk_id, value)
        return
    if not _local_active:
        raise RuntimeError("relay.save_result() must be called inside a relay.chunks() loop")
    _local_results[_local_active[-1]][chunk_id] = value


def load_results(name: str | None = None) -> list[Any]:
    """All saved results of the most recent (or named) chunks loop, ordered by chunk id."""
    if _session is not None:
        return _session.load_results(name)
    bucket = name or (_local_last[0] if _local_last else None)
    if bucket is None or bucket not in _local_results:
        raise RuntimeError("relay.load_results(): no relay.chunks() loop has run yet")
    results = _local_results[bucket]
    return [results[k] for k in sorted(results)]


def safe_point() -> None:
    """Optional extra safe point. On resume, execution continues right after it."""
    if _session is not None:
        _session.safe_point()
