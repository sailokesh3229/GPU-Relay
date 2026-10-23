"""The in-session runtime object behind instrumentation, framework hooks, and the SDK.

One :class:`RelaySession` exists per segment. It owns checkpointing, output capture,
working-file tracking, the status heartbeat, the watchdog, and the position tracker.

What a checkpoint captures (spec §7.3, §7.5) for a script whose top-level statements
``0..P-1`` form the *replay prefix* (cheap, side-effect-free setup re-executed on resume):

* every name assigned or mutated by statements ``P..current`` (they are skipped on resume),
* names passed as call arguments in those statements when bound to mutable containers or
  instances of classes defined in the script,
* names declared ``global`` inside functions,
* every torch-stateful object in the namespace,
* everything registered with ``relay.state`` (stored under ``relay:<name>``).
"""

from __future__ import annotations

import base64
import builtins
import collections
import contextlib
import itertools
import json
import logging
import pickle
import random
import sys
import time
import types
from collections.abc import Callable, Iterable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle

from gpurelay.config import RuntimeSettings
from gpurelay.errors import StateRestoreError
from gpurelay.models import ExitCode, ParallelStatus, Progress, Rate, RuntimeState
from gpurelay.runtime.blobs import BlobCache
from gpurelay.runtime.capture import LOG_ENCODING, OutputCapture
from gpurelay.runtime.checkpoint import CheckpointManager, LoadedCheckpoint
from gpurelay.runtime.instrument import ScriptPlan, StatementInfo, called_functions
from gpurelay.runtime.loops import (
    FrameworkPlan,
    Position,
    ResumeMismatchError,
    Tracker,
    run_loop,
)
from gpurelay.runtime.parallel import ParallelRegion, inside_region, region_id
from gpurelay.runtime.persist import PersistResult
from gpurelay.runtime.rng import capture_rng, chunk_seed, load_rng, restore_rng, seed_all
from gpurelay.runtime.state import (
    RestoreReport,
    is_stateful_closure,
    is_torch_stateful,
    is_user_object,
    restore_state,
    skip_reason,
)
from gpurelay.runtime.status import StatusSnapshot, StatusWriter
from gpurelay.runtime.watchdog import Watchdog
from gpurelay.runtime.workfiles import WorkFileTracker
from gpurelay.sdk import SdkLoop
from gpurelay.storage.base import Storage
from gpurelay.storage.layout import control_key, segment_log_key, worker_status_key

log = logging.getLogger(__name__)

REGISTRY_PREFIX = "relay:"
LOCAL_PREFIX = "local:"  # "local:<loop key>|<name>": a local of a function with a loop
FRAMEWORK_DIR = "framework"


class Handoff(BaseException):
    """Raised at a safe point after the handoff checkpoint committed (exit 85).

    A ``BaseException`` so that ``except Exception`` in user code does not swallow it.
    """


class WorkerDone(Handoff):
    """A parallel-chunks worker has nothing left to compute (exit 0, nothing checkpointed)."""


class RelayAbort(BaseException):
    """Runtime failure surfaced through user code (exit code attached)."""

    def __init__(self, exit_code: ExitCode, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class AutoApi:
    """The ``__relay__`` object that instrumented code calls."""

    def __init__(self, session: RelaySession) -> None:
        self._session = session

    def loop(self, loop_id: str, iterable: Any, local_names: tuple[str, ...] | None = None) -> Any:
        """``for`` loop; ``local_names`` is passed for loops inside functions (Phase 9)."""
        frame = sys._getframe(1) if local_names is not None else None
        return self._session.auto_loop(loop_id, iterable, frame, local_names or ())

    def forever(self) -> Iterator[int]:
        """Iterable of an instrumented ``while`` loop (``itertools.count``)."""
        return itertools.count()


def apply_in_place(live: Any, new: Any) -> Any:
    """Copy ``new`` into ``live`` when it is a mutable container; return the object to keep.

    Keeps other references to ``live`` valid (``history = []; relay.state(history=history)``).
    """
    if live is new or live is None:
        return new
    if type(live) is type(new) and is_rng_object(live):
        _copy_rng_state(live, new)  # e.g. a DataLoader holds the live generator
        return live
    if (
        is_stateful_closure(live)
        and is_stateful_closure(new)
        and live.__code__.co_code == new.__code__.co_code
        and len(live.__closure__) == len(new.__closure__)
    ):
        # Refill the live cells: functions sharing them (same factory call) stay in sync.
        for live_cell, new_cell in zip(live.__closure__, new.__closure__, strict=True):
            with contextlib.suppress(ValueError):  # an empty cell
                live_cell.cell_contents = new_cell.cell_contents
        return live
    if isinstance(live, list) and isinstance(new, list):
        live[:] = new
        return live
    if isinstance(live, dict) and isinstance(new, dict):
        live.clear()
        live.update(new)
        return live
    if isinstance(live, set) and isinstance(new, set):
        live.clear()
        live.update(new)
        return live
    if isinstance(live, bytearray) and isinstance(new, bytearray | bytes):
        live[:] = new
        return live
    np = sys.modules.get("numpy")
    if (
        np is not None
        and isinstance(live, np.ndarray)
        and isinstance(new, np.ndarray)
        and live.shape == new.shape
        and live.dtype == new.dtype
        and live.flags.writeable
    ):
        live[...] = new
        return live
    torch = sys.modules.get("torch")
    if (
        torch is not None
        and isinstance(live, torch.Tensor)
        and isinstance(new, torch.Tensor)
        and live.shape == new.shape
        and live.dtype == new.dtype
    ):
        with torch.no_grad():
            live.copy_(new)
        return live
    return new


_CONTAINERS = (list, dict, set, bytearray, collections.deque)


def is_rng_object(value: Any) -> bool:
    """A random generator object (``random.Random``, NumPy ``Generator``/``RandomState``,
    ``torch.Generator``). Always snapshotted: code advances it without reassigning it
    (``rng.random()``, ``randint(..., generator=g)``), so replaying the statement that
    created it would restart its stream."""
    if isinstance(value, random.Random):
        return True
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.random.Generator | np.random.RandomState):
        return True
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Generator)


def _copy_rng_state(live: Any, new: Any) -> None:
    if isinstance(live, random.Random):
        live.setstate(new.getstate())
    elif hasattr(live, "bit_generator"):  # numpy Generator
        live.bit_generator.state = new.bit_generator.state
    else:  # numpy RandomState and torch.Generator
        live.set_state(new.get_state())


def _is_mutable_arg(value: Any) -> bool:
    """Heuristic for names passed to functions that may mutate them in place."""
    return isinstance(value, _CONTAINERS) or type(value).__module__ == "__main__"


def _safe_len(iterable: Any) -> int | None:
    try:
        return len(iterable)
    except Exception:
        return None


def _indexable(iterable: Any) -> bool:
    if isinstance(iterable, Sequence):
        return True
    np = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    return (np is not None and isinstance(iterable, np.ndarray)) or (
        torch is not None and isinstance(iterable, torch.Tensor)
    )


@dataclass
class ScriptState:
    plan: ScriptPlan
    namespace: dict[str, Any]


class RelaySession:
    """Runtime state for one segment of one job."""

    def __init__(
        self,
        *,
        storage: Storage,
        job_id: str,
        segment_id: str,
        work_dir: Path,
        runtime_dir: Path,
        settings: RuntimeSettings,
        deadline: float | None,
        loaded: LoadedCheckpoint | None,
        baseline_files: dict[str, str] | None = None,
        exclude_files: set[str] | None = None,
        clock: Callable[[], float] = time.time,
        worker_region: str | None = None,
    ) -> None:
        self.storage = storage
        # Worker segments only compute chunks of one parallel region: no checkpoints, no
        # job status, no outputs of their own (Phase 9, ``runtime/parallel.py``).
        self.worker_region = worker_region
        self.job_id = job_id
        self.segment_id = segment_id
        self.work_dir = Path(work_dir)
        self.runtime_dir = Path(runtime_dir)
        self.settings = settings
        self.loaded = loaded
        self.clock = clock

        meta = loaded.runtime if loaded else {}
        self.base_chain: list[dict[str, Any]] = list(meta.get("chain", []))
        self.files_manifest: dict[str, str] = dict(meta.get("files", {}))
        self.results_index: dict[str, dict[str, str]] = {
            bucket: dict(entries) for bucket, entries in meta.get("results", {}).items()
        }
        self.prefix_end: int | None = meta.get("prefix_end")
        self.durations: dict[int, float] = {
            int(k): float(v) for k, v in meta.get("durations", {}).items()
        }
        self.torch_ctors: set[int] = set(meta.get("torch_ctors", []))
        # RNG state at the start of active loops over sampler-driven iterables (DataLoader):
        # restored before the iterator is recreated on resume, so batches come in the same
        # order (mid-epoch resume, Phase 9).
        self._iter_rng: dict[str, Any] = {}
        self._resume_iter_rng: dict[str, Any] = {
            key: pickle.loads(base64.b64decode(raw))
            for key, raw in meta.get("iter_rng", {}).items()
        }
        self.resume_target: Position | None = loaded.positions.get("position") if loaded else None

        self.blobs = BlobCache(self.runtime_dir / "blobs", storage, job_id)
        self.files = WorkFileTracker(self.work_dir, self.blobs, baseline_files, exclude_files)
        self.capture = OutputCapture(
            self.runtime_dir / "outputs" / "log.txt", self.blobs, recording=loaded is None
        )
        self.capture.warnings_shown.update(meta.get("warnings_shown", []))
        self.manager = CheckpointManager(
            storage,
            job_id=job_id,
            segment_id=segment_id,
            local_root=self.runtime_dir / "ckpt",
            zstd_level=settings.zstd_level,
            keep=settings.keep_checkpoints,
        )
        self.manager.mark_blobs_known(self._referenced_blobs(self.base_chain))
        self.watchdog = Watchdog(settings, deadline, clock)
        self.tracker = Tracker(on_reach=self._on_reach)
        self.status = StatusWriter(
            storage,
            job_id,
            segment_id,
            settings.heartbeat_s,
            self._status_snapshot,
            on_tick=self._poll_control,
            key=worker_status_key(job_id, segment_id) if worker_region else None,
        )

        self.script: ScriptState | None = None
        self.registry: dict[str, Any] = {}
        self._pending: dict[str, Any] = {}
        # Restored on resume: a resume past a chunks loop skips the loop statement, and
        # relay.load_results() after it still needs to know which bucket it filled.
        self._last_bucket: str | None = meta.get("last_bucket")
        self._chunk_buckets: list[str] = []
        self._named_buckets: dict[str, str] = dict(meta.get("named_buckets", {}))
        self._frameworks: list[FrameworkPlan] = []
        self._warned_unpicklable: set[str] = set()
        # Extension points (notebook executor): extra checkpoint metadata + blobs, and
        # callbacks run when the resume point is reached.
        self.runtime_extras: list[Callable[[], tuple[dict[str, Any], set[str]]]] = []
        self.on_restore: list[Callable[[], None]] = []
        self._live_since: float | None = None if loaded else clock()
        self._live_iterations = 0
        self._rate_origin: dict[str, tuple[int, float]] = {}
        # Parallel chunks regions (Phase 9).
        self._force_checkpoint = False
        self._last_position: Position | None = None
        self._region: tuple[ParallelRegion, int, bool] | None = None  # region, total, ready
        self._region_entry_seq: int | None = None
        self._claim: tuple[ParallelRegion, int] | None = None
        self._pending_mark: dict[str, Any] | None = None
        # Frames of active instrumented loops inside functions: loop key -> (frame, names).
        self._frames: dict[str, tuple[types.FrameType, tuple[str, ...]]] = {}
        self.reached_resume_point = loaded is None
        self.relay_api = AutoApi(self)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self.loaded is not None:
            # Restore files before replay so code that reads them works (spec §7.5).
            self.files.materialize(self.files_manifest)
        self.files.scan()
        self.capture.start()
        self.watchdog.install_signal_handlers()
        self.watchdog.reset_timer()
        self.status.start()

    def close(self, state: RuntimeState, message: str = "") -> None:
        """Stop background activity and publish the final status."""
        try:
            self.capture.stop()
            self.watchdog.uninstall_signal_handlers()
            result = self.manager.close()
            self.watchdog.record_persist(result)
        finally:
            self.status.stop()
            self.status.write(state, message)

    def attach_script(self, plan: ScriptPlan, namespace: dict[str, Any]) -> None:
        self.script = ScriptState(plan, namespace)

    # ------------------------------------------------------------------ statements

    def run_statement(
        self,
        stmt: StatementInfo,
        *,
        target: Position | None = None,
        passthrough: bool = False,
        execute: Callable[[StatementInfo], None] | None = None,
    ) -> None:
        """Execute one top-level statement (or notebook cell) of the attached script."""
        assert self.script is not None
        self.tracker.begin_statement(stmt.index, target=target, passthrough=passthrough)
        if execute is not None:
            execute(stmt)
        else:
            assert stmt.code is not None
            exec(stmt.code, self.script.namespace)
        self.tracker.end_statement()

    def run_definitions(self, stmt: StatementInfo) -> None:
        """Re-create the imports/functions/classes of a skipped cell (output suppressed)."""
        assert self.script is not None
        if stmt.definitions_code is None:
            return
        self.tracker.begin_statement(stmt.index, passthrough=True)
        exec(stmt.definitions_code, self.script.namespace)
        self.tracker.end_statement()

    def statement_done(self, stmt: StatementInfo, duration_s: float | None) -> None:
        """Record replayability, then offer a statement-boundary safe point."""
        assert self.script is not None
        if duration_s is not None:
            self.durations[stmt.index] = duration_s
        if self.prefix_end is None and not self.replayable(stmt, duration_s):
            self.prefix_end = stmt.index
        self.on_safe_point({"stmt": stmt.index, "end": "stmt"})

    def replayable(self, stmt: StatementInfo, duration_s: float | None) -> bool:
        """Can ``stmt`` be re-executed cheaply and safely on resume?"""
        assert self.script is not None
        if stmt.is_definition or stmt.is_setup:
            return True
        if duration_s is None or duration_s >= self.settings.replay_threshold_s:
            return False
        if stmt.side_effects:
            return False
        return not any(f.side_effects for f in called_functions(stmt, self.script.plan))

    def should_replay(self, stmt: StatementInfo) -> bool:
        """Re-execute ``stmt`` (before the resume point) as setup?"""
        return (
            stmt.is_definition
            or stmt.is_setup
            or self.prefix_end is None
            or stmt.index < self.prefix_end
        )

    def should_replay_constructor(self, stmt: StatementInfo) -> bool:
        """Re-execute a skipped statement that built torch objects (after restoring values)?

        Rebuilding e.g. ``optimizer = Adam(model.parameters())`` around the live model keeps
        object references intact; the snapshot's state is then loaded into the new object.
        Only loop-free, side-effect-free statements that ran reasonably fast qualify.
        """
        assert self.script is not None
        if stmt.index not in self.torch_ctors or stmt.loops or stmt.uninstrumented:
            return False
        if stmt.side_effects or any(
            f.side_effects or f.loops for f in called_functions(stmt, self.script.plan)
        ):
            return False
        limit = 10 * self.settings.replay_threshold_s
        return self.durations.get(stmt.index, float("inf")) < limit

    # ------------------------------------------------------------------ resume

    def restore_values(self) -> None:
        """Put the snapshot's variables into the namespace (no RNG, still replaying).

        Done when the target statement starts, so code before the resume point inside it
        sees the results of statements that were skipped.
        """
        if self.loaded is None or self.script is None:
            return
        try:
            self._apply_snapshot()
        except StateRestoreError as exc:
            raise RelayAbort(ExitCode.NON_RESUMABLE, f"cannot restore state: {exc}") from exc

    def reach_boundary(self) -> None:
        """Resume point is a statement boundary: restore and go live."""
        self._on_reach()

    def _on_reach(self) -> None:
        """The resume point was reached: restore state, RNG, files; go live."""
        try:
            self._restore()
        except StateRestoreError as exc:
            raise RelayAbort(ExitCode.NON_RESUMABLE, f"cannot restore state: {exc}") from exc
        except Exception as exc:
            raise RelayAbort(ExitCode.INFRA_ERROR, f"restore failed: {exc!r}") from exc

    def _apply_snapshot(self) -> RestoreReport:
        assert self.loaded is not None
        ns = self.script.namespace if self.script is not None else {}
        combined: dict[str, Any] = dict(ns)
        for name, value in self.registry.items():
            combined[REGISTRY_PREFIX + name] = value
        frames = dict(self._frames)
        for key, (frame, _names) in frames.items():
            for name, value in dict(frame.f_locals).items():
                combined[f"{LOCAL_PREFIX}{key}|{name}"] = value
        report = restore_state(combined, self.loaded.local_dir)
        local_updates: dict[str, dict[str, Any]] = {}
        for name in report.unpickled_torch + report.plain:
            new = combined[name]
            if name.startswith(LOCAL_PREFIX):
                key, _, local = name[len(LOCAL_PREFIX) :].rpartition("|")
                if key in frames:  # the function is running (resume point inside it)
                    live = dict(frames[key][0].f_locals).get(local)
                    local_updates.setdefault(key, {})[local] = apply_in_place(live, new)
                continue
            if name.startswith(REGISTRY_PREFIX):
                key = name[len(REGISTRY_PREFIX) :]
                if key in self.registry:
                    self.registry[key] = apply_in_place(self.registry[key], new)
                else:
                    self._pending[key] = new
            elif self.script is not None:
                ns[name] = apply_in_place(ns.get(name), new)
        if self.script is not None:
            for name in report.imported:
                if not name.startswith(LOCAL_PREFIX):
                    ns[name] = combined[name]
        for key, values in local_updates.items():
            set_frame_locals(frames[key][0], values)
        return report

    def _restore(self) -> None:
        assert self.loaded is not None
        report = self._apply_snapshot()
        warnings = report.warnings + load_rng(self.loaded.local_dir / "rng.pt")
        if report.unpickled_torch and report.loaded_into_live:
            warnings.append(
                f"torch objects {report.unpickled_torch} were restored from the checkpoint "
                f"while {report.loaded_into_live} were loaded into live objects; references "
                "between the two groups are not shared (e.g. an optimizer may not drive the "
                "live model)"
            )
        for message in warnings:
            log.warning(message)
        self.files.revert(self.files_manifest)
        for callback in self.on_restore:
            callback()
        self.capture.set_recording(True)
        self.watchdog.reset_timer()
        self._live_since = self.clock()
        self.reached_resume_point = True
        self.resume_target = None
        self.status.write()
        log.info("resumed from checkpoint %d", self.loaded.seq)

    # ------------------------------------------------------------------ loops

    def auto_loop(
        self,
        loop_id: str,
        iterable: Any,
        frame: types.FrameType | None = None,
        local_names: tuple[str, ...] = (),
    ) -> Any:
        """Instrumented ``for`` loop (Layer C); ``frame``: the function running it, if any."""
        if isinstance(iterable, SdkLoop) or self.tracker.passthrough or self.tracker.untracked:
            return iterable
        if frame is not None:
            key = self.tracker.next_loop_key()
            loop = self._auto_loop(loop_id, iterable)
            return self._with_frame(key, frame, local_names, loop)
        return self._auto_loop(loop_id, iterable)

    def _with_frame(
        self, key: str, frame: types.FrameType, names: tuple[str, ...], loop: Iterator[Any]
    ) -> Iterator[Any]:
        """While the loop runs, its function's locals are part of every snapshot."""
        self._frames[key] = (frame, names)
        try:
            yield from loop
        finally:
            self._frames.pop(key, None)

    def _auto_loop(self, loop_id: str, iterable: Any) -> Any:
        total = _safe_len(iterable)
        if isinstance(iterable, MutableSequence):
            # Python iterates a list by index while index < len(list): it may grow or shrink
            # during the loop, so the length is read at every step (and unknown up front).
            total = None

            def items_from(start: int) -> Iterator[tuple[int, Any]]:
                index = start
                while index < len(iterable):
                    yield index, iterable[index]
                    index += 1

        elif total is not None and _indexable(iterable):

            def items_from(start: int) -> Iterator[tuple[int, Any]]:
                return ((i, iterable[i]) for i in range(start, total))

        else:
            key = self.tracker.next_loop_key()
            sampled = _sampler_driven(iterable)

            def items_from(start: int) -> Iterator[tuple[int, Any]]:
                if sampled:
                    saved = self._resume_iter_rng.get(key) if self.tracker.replaying else None
                    if saved is not None:
                        restore_rng(saved)  # same shuffle as in the checkpointed run
                    self._iter_rng[key] = saved if saved is not None else capture_rng()
                it = iter(iterable)
                if start:
                    began = time.perf_counter()
                    if _skip_batches(it, start):
                        how = "skipping (without loading)"
                    else:
                        how = "consuming"
                        start = sum(1 for _ in itertools.islice(it, start))
                    log.info("fast-forwarded loop %s by %s %d item(s) in %.1fs", loop_id, how,
                             start, time.perf_counter() - began)  # fmt: skip
                return ((start + n, item) for n, item in enumerate(it))

            if sampled:
                return self._forget_iter_rng(key, run_loop(
                    self.tracker, total, loop_id, items_from, self._on_iteration_end,
                    loop_id=loop_id, adaptive=True,
                ))  # fmt: skip

        return run_loop(
            self.tracker, total, loop_id, items_from, self._on_iteration_end, loop_id=loop_id,
            adaptive=True,
        )  # fmt: skip

    def _forget_iter_rng(self, key: str, loop: Iterator[Any]) -> Iterator[Any]:
        try:
            yield from loop
        finally:
            self._iter_rng.pop(key, None)

    def loop(
        self,
        total: int | None,
        name: str | None,
        items_from: Callable[[int], Iterator[tuple[int, Any]]],
        *,
        chunk_loop: bool = False,
    ) -> Iterator[Any]:
        """SDK loop; ``items_from(start)`` yields ``(index, item)`` from ``start``."""
        if not chunk_loop:
            yield from run_loop(self.tracker, total, name, items_from, self._on_iteration_end)
            return
        # Results bucket: the loop's position key (stable across segments) or its name.
        bucket = name or self.tracker.next_loop_key()
        self._last_bucket = bucket
        if name:
            self._named_buckets[name] = bucket
        self._chunk_buckets.append(bucket)
        try:
            yield from run_loop(self.tracker, total, name, items_from, self._on_iteration_end)
        finally:
            self._chunk_buckets.remove(bucket)

    def parallel_chunks(
        self,
        total: int,
        name: str | None,
        chunk_at: Callable[[int], Any],
        seed: int,
    ) -> Iterator[Any]:
        """``relay.chunks(..., parallel=True)``: chunks shared with worker segments."""
        bucket = name or self.tracker.next_loop_key()
        rid = region_id(bucket)
        live_worker = self.worker_region is not None and self.reached_resume_point
        if live_worker and self.worker_region != rid:
            raise WorkerDone()  # a worker never runs code outside its region
        info = {
            "bucket": bucket,
            "stmt": self.tracker.stmt,
            "address": [list(step) for step in self.tracker.current.address],
            "event": self.tracker.current.events,
            "total": total,
        }
        region = ParallelRegion(
            self.storage, self.job_id, rid, self.segment_id, self.settings.stale_heartbeat_s
        )
        worker = self.worker_region == rid
        resumed_inside = inside_region(self.resume_target, info)

        def sequential(start: int) -> Iterator[tuple[int, Any]]:
            for k in builtins.range(start, total):
                seed_all(chunk_seed(seed, k))
                yield k, chunk_at(k)

        def main_items(start: int) -> Iterator[tuple[int, Any]]:
            announced = False
            for k in builtins.range(start, total):
                if not self.reached_resume_point:  # replaying into a chunk (deeper target)
                    seed_all(chunk_seed(seed, k))
                    yield k, chunk_at(k)
                    continue
                if not announced:
                    announced = True
                    region.write_info(info)
                    self._region = (region, total, resumed_inside)
                    self._region_entry_seq = self.manager.last_committed_seq
                    self._force_checkpoint = True  # workers start from a checkpoint in here
                marker = self._wait_for_chunk(region, k)
                if marker is not None:
                    self._merge_chunk(bucket, k, marker)
                    continue
                yield from self._compute_chunk(region, bucket, k, chunk_at, seed, output=False)

        def worker_items(_start: int) -> Iterator[tuple[int, Any]]:
            for k in builtins.range(total - 1, -1, -1):
                if self.watchdog.handoff_due():
                    raise WorkerDone()
                if region.done(k) is not None or region.claimed_by_other(k):
                    continue
                yield from self._compute_chunk(region, bucket, k, chunk_at, seed, output=True)

        self._last_bucket = bucket
        if name:
            self._named_buckets[name] = bucket
        self._chunk_buckets.append(bucket)
        try:
            if self.tracker.passthrough:
                items: Callable[[int], Iterator[tuple[int, Any]]] = sequential
            else:
                items = worker_items if worker else main_items
            yield from run_loop(self.tracker, total, name, items, self._on_iteration_end)
            if worker and self.reached_resume_point:
                raise WorkerDone()
        finally:
            self._chunk_buckets.remove(bucket)
            self._region = None
            self._claim = None
            self._pending_mark = None
        seed_all(chunk_seed(seed, total))  # same RNG state after the loop on every path

    def _compute_chunk(
        self,
        region: ParallelRegion,
        bucket: str,
        k: int,
        chunk_at: Callable[[int], Any],
        seed: int,
        *,
        output: bool,
    ) -> Iterator[tuple[int, Any]]:
        region.claim(k)
        self._claim = (region, k)
        self.capture.flush()
        self._pending_mark = {
            "region": region,
            "bucket": bucket,
            "k": k,
            "offset": self.capture.offset if output else None,
            "stmt": self.tracker.stmt,
            "depth": len(self.tracker.current.address),
            "event": self.tracker.current.events - 1,  # the event of the running loop
        }
        seed_all(chunk_seed(seed, k))
        yield k, chunk_at(k)

    def _publish_chunk(self, pending: dict[str, Any]) -> None:
        """Chunk finished: publish its result (and, from workers, its printed output)."""
        region: ParallelRegion = pending["region"]
        k: int = pending["k"]
        sha = self.results_index.get(pending["bucket"], {}).get(str(k), "")
        result = (sha, self.blobs.read_bytes(sha)) if sha else ("", b"")
        out = ("", b"")
        if pending["offset"] is not None:
            self.capture.flush()
            with open(self.capture.log_path, "rb") as fh:
                fh.seek(pending["offset"])
                data = fh.read()
            if data:
                out = (self.blobs.put_bytes(data), data)
        try:
            region.mark_done(k, result, out)
        except Exception as exc:
            log.warning("could not publish chunk %d: %s", k, exc)  # recomputed elsewhere
        else:
            self.manager.mark_blobs_known({s for s in (result[0], out[0]) if s})
        self._claim = None

    def _merge_chunk(self, bucket: str, k: int, marker: dict[str, Any]) -> None:
        """Chunk ``k`` was computed by another session: take its result and output."""
        if marker.get("result"):
            self.results_index.setdefault(bucket, {})[str(k)] = marker["result"]
        known = {s for s in (marker.get("result"), marker.get("output")) if s}
        self.manager.mark_blobs_known(known)
        if marker.get("output"):
            text = self.blobs.read_bytes(marker["output"]).decode(LOG_ENCODING)
            sys.stdout.write(text)  # into the log in chunk order, as if computed here

    def _wait_for_chunk(self, region: ParallelRegion, k: int) -> dict[str, Any] | None:
        """Marker of chunk ``k`` if done; waits while another live session computes it."""
        poll_s = min(max(self.settings.heartbeat_s / 2, 1.0), 10.0)
        while True:
            marker = region.done(k)
            if marker is not None or not region.claimed_by_other(k):
                return marker
            if self.watchdog.handoff_due():
                if self._last_position is not None:
                    self._handoff(self._last_position)
                raise Handoff()  # nothing new since the checkpoint this segment resumed from
            time.sleep(poll_s)

    def _on_iteration_end(self, position: Position) -> None:
        self._live_iterations += 1
        if self._pending_mark is not None and _is_iteration_of(position, self._pending_mark):
            self._publish_chunk(self._pending_mark)
            self._pending_mark = None
        self.on_safe_point(position)

    def register(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if name in self._pending:
                value = apply_in_place(value, self._pending.pop(name))
            self.registry[name] = value

    def safe_point(self) -> None:
        position = self.tracker.safe_point()
        if position is not None:
            self.on_safe_point(position)

    def on_safe_point(self, position: Position) -> None:
        if self.tracker.passthrough or self.worker_region is not None:
            return
        self._last_position = position
        self.watchdog.note_safe_point()
        try:
            if self.watchdog.handoff_due():
                self._handoff(position)
            if self.watchdog.checkpoint_due() or self._force_checkpoint:
                self._force_checkpoint = False
                self.checkpoint(position)
        except Exception as exc:
            # Runtime failures must not look like user-code errors.
            raise RelayAbort(ExitCode.INFRA_ERROR, f"checkpoint failed: {exc!r}") from exc

    def _handoff(self, position: Position, extra: dict[str, Path] | None = None) -> None:
        result = self.checkpoint(position, wait=True, extra=extra)
        if result is not None and result.ok:
            raise Handoff()
        error = result.error if result is not None else None
        raise RelayAbort(ExitCode.INFRA_ERROR, f"handoff checkpoint failed: {error}")

    # ------------------------------------------------------------------ frameworks (Layer B)

    def framework_call(self, kind: str, run: Callable[[Path | None], Any]) -> Any:
        """Run a framework training call (``Trainer.train``/``fit``) as a resumable event.

        ``run(resume_path)`` starts training, resuming from ``resume_path`` when given.
        """
        plan = self.tracker.framework_start(kind)
        if plan.passthrough:
            return run(None)
        if plan.skip:
            log.warning("skipping %s call that completed before the checkpoint", kind)
            return None
        resume = None
        if plan.resume:
            assert self.loaded is not None
            resume = self.loaded.local_dir / FRAMEWORK_DIR
            if not resume.exists():
                raise RelayAbort(ExitCode.NON_RESUMABLE, f"{kind} checkpoint missing on resume")
        self._frameworks.append(plan)
        try:
            return run(resume)
        finally:
            self._frameworks.pop()

    def reapply_rng(self) -> None:
        """Set RNG states to the checkpoint's again (framework setup may have used RNG
        between the resume point and the first resumed epoch)."""
        if self.loaded is not None:
            for message in load_rng(self.loaded.local_dir / "rng.pt"):
                log.warning(message)

    def framework_due(self, lookahead_s: float | None = None) -> bool:
        """Should the active framework save a checkpoint now?

        ``lookahead_s`` is the time until the framework's next acceptable save point; a
        handoff is requested early enough that the next one would not be too late.
        """
        if not self._frameworks or self.tracker.passthrough or self.tracker.replaying:
            return False
        self.watchdog.note_safe_point()  # framework step/epoch boundary: gap bookkeeping
        return self.watchdog.handoff_due(lookahead_s) or self.watchdog.checkpoint_due()

    def framework_safe_point(self, framework_state: Path, lookahead_s: float | None = None) -> None:
        """The framework just saved ``framework_state``: snapshot (and maybe hand off)."""
        if not self._frameworks:
            return
        position = self.tracker.framework_position(self._frameworks[-1])
        extra = {FRAMEWORK_DIR: Path(framework_state)}
        try:
            if self.watchdog.handoff_due(lookahead_s):
                self._handoff(position, extra)
            self.checkpoint(position, extra=extra)
        except Exception as exc:
            raise RelayAbort(ExitCode.INFRA_ERROR, f"checkpoint failed: {exc!r}") from exc

    # ------------------------------------------------------------------ results

    def save_result(self, chunk_id: int, value: Any) -> None:
        if not self._chunk_buckets and not self.tracker.passthrough:
            raise RuntimeError("relay.save_result() must be called inside a relay.chunks() loop")
        if not self.reached_resume_point or self.tracker.passthrough:
            return  # replayed work before the resume point is discarded
        sha = self.blobs.put_bytes(cloudpickle.dumps(value))
        self.results_index.setdefault(self._chunk_buckets[-1], {})[str(chunk_id)] = sha

    def load_results(self, name: str | None = None) -> list[Any]:
        bucket = self._named_buckets.get(name, name) if name else self._last_bucket
        if bucket is None:
            raise RuntimeError("relay.load_results(): no relay.chunks() loop has run yet")
        entries = self.results_index.get(bucket, {})
        return [
            cloudpickle.loads(self.blobs.read_bytes(entries[key]))
            for key in sorted(entries, key=int)
        ]

    # ------------------------------------------------------------------ checkpoint

    def _referenced_blobs(self, chain: Iterable[dict[str, Any]]) -> set[str]:
        shas = set(self.files_manifest.values())
        for entry in chain:
            shas.update(entry.get("figures", []))
        for entries in self.results_index.values():
            shas.update(entries.values())
        return shas

    def _snapshot_namespace(
        self, position: Position
    ) -> tuple[dict[str, Any], int | None, list[int]]:
        """Variables to capture at ``position``, the effective replay-prefix end, and the
        skipped statements that constructed torch objects."""
        selected: dict[str, Any] = {
            REGISTRY_PREFIX + name: value for name, value in self.registry.items()
        }
        if self.script is None or position.get("end") == "final":
            return selected, self.prefix_end, []
        ns, plan = self.script.namespace, self.script.plan
        current = int(position["stmt"])
        completed = position["end"] == "stmt"
        if self.prefix_end is not None:
            p_eff = self.prefix_end
        else:
            p_eff = current + 1 if completed else current
        names = set(plan.global_names)
        arg_names: set[str] = set()
        receivers: set[str] = set()
        for stmt in plan.statements[p_eff : current + 1]:
            if not stmt.is_definition:
                names |= stmt.assigned
                arg_names |= stmt.call_args
                receivers |= stmt.receivers
        for name in receivers - names:  # tracker.record(x): user objects change via methods
            if name in ns and is_user_object(ns[name]):
                selected[name] = ns[name]
        for name in names:
            if name in ns:
                selected[name] = ns[name]
        for name in arg_names - names:
            if name in ns and _is_mutable_arg(ns[name]):
                selected[name] = ns[name]
        for name, value in ns.items():
            if (
                name not in selected
                and skip_reason(name, value) is None
                and (is_torch_stateful(value) or is_rng_object(value) or is_stateful_closure(value))
            ):
                selected[name] = value
        for key, (frame, local_names) in self._frames.items():
            local_ns = dict(frame.f_locals)
            wanted = set(local_names)
            for name, value in local_ns.items():
                if skip_reason(name, value) is None and (
                    name in wanted
                    or is_torch_stateful(value)
                    or is_rng_object(value)
                    or is_stateful_closure(value)
                    or is_user_object(value)
                ):
                    selected[f"{LOCAL_PREFIX}{key}|{name}"] = value
        last_skipped = current if completed else current - 1
        ctors = sorted(
            {j for j in self.torch_ctors if j < p_eff}
            | {
                stmt.index
                for stmt in plan.statements[p_eff : last_skipped + 1]
                if any(n in ns and is_torch_stateful(ns[n]) for n in stmt.assigned)
            }
        )
        return selected, p_eff, ctors

    def checkpoint(
        self,
        position: Position,
        *,
        final: bool = False,
        wait: bool = False,
        extra: dict[str, Path] | None = None,
    ) -> PersistResult | None:
        """Snapshot now (at a safe point) and persist in the background."""
        if self.worker_region is not None:
            return None  # workers never checkpoint: the main session owns the job state
        self.capture.flush()
        log_bytes = self.capture.offset
        chain = [
            *self.base_chain,
            {
                "segment": self.segment_id,
                "log_bytes": log_bytes,
                "figures": list(self.capture.figures),
            },
        ]
        self.files_manifest = self.files.scan()
        log_copy = self.runtime_dir / "outputs" / "log-snapshot.txt"
        self.capture.copy_log(log_copy, log_bytes)
        extra_meta: dict[str, Any] = {}
        extra_blobs: set[str] = set()
        for provider in self.runtime_extras:
            meta_part, shas = provider()
            extra_meta.update(meta_part)
            extra_blobs |= shas
        blobs = {sha: self.blobs.path(sha) for sha in self._referenced_blobs(chain) | extra_blobs}
        namespace, p_eff, ctors = self._snapshot_namespace(position)
        info = self.manager.snapshot(
            namespace,
            safe_point_id=json.dumps(position, separators=(",", ":")),
            positions={"position": position},
            runtime={
                "chain": chain,
                "files": self.files_manifest,
                "results": self.results_index,
                "last_bucket": self._last_bucket,
                "named_buckets": self._named_buckets,
                "prefix_end": self.prefix_end,
                "snapshot_prefix": p_eff,
                "durations": {str(k): v for k, v in sorted(self.durations.items())},
                "torch_ctors": ctors,
                "warnings_shown": sorted(self.capture.warnings_shown),
                "iter_rng": {
                    key: base64.b64encode(pickle.dumps(state)).decode("ascii")
                    for key, state in self._iter_rng.items()
                },
                "final": final,
                **extra_meta,
            },
            extra=extra,
            uploads={segment_log_key(self.job_id, self.segment_id): log_copy},
            blobs=blobs,
        )
        for name, why in info.state.unpicklable.items():
            if name not in self._warned_unpicklable:
                self._warned_unpicklable.add(name)
                log.warning("variable %r is not picklable and is not checkpointed: %s", name, why)
        self.watchdog.record_snapshot(info)
        self.watchdog.record_persist(self.manager.stats.last_persist)
        if not wait:
            return None
        result = self.manager.wait()
        self.watchdog.record_persist(result)
        return result

    def finish(self) -> PersistResult | None:
        """Program ended normally: final checkpoint of outputs (spec §12.1 step 7)."""
        if self.worker_region is not None:
            raise WorkerDone()
        if self.resume_target is not None:
            raise ResumeMismatchError("program ended before reaching the resume point")
        return self.checkpoint({"end": "final"}, final=True, wait=True)

    # ------------------------------------------------------------------ control

    def _poll_control(self) -> None:
        """``jobs/<job>/control.json`` written by the controller: pause = hand off now."""
        claim = self._claim
        if claim is not None:
            claim[0].claim(claim[1])  # keep the claim visibly alive while computing
        raw = self.storage.get_bytes(control_key(self.job_id))
        if not raw:
            return
        try:
            control = json.loads(raw)
        except ValueError:
            return
        if control.get("handoff") and control.get("segment") in (None, self.segment_id):
            if not self.watchdog.stop_requested:
                log.info("controller requested a handoff (%s)", control.get("reason", "pause"))
            self.watchdog.stop_requested = True

    # ------------------------------------------------------------------ status

    def _status_snapshot(self) -> StatusSnapshot:
        progress = Progress(cell=self.tracker.stmt, loop=self.tracker.progress())
        rate = None
        outer = self.tracker.active[0] if self.tracker.active else None
        if outer is not None and self.reached_resume_point:
            # Rate of the outermost loop (what the estimator divides remaining work by),
            # measured from the first time this segment saw the loop live.
            now = self.clock()
            first = self._rate_origin.get(outer.key)
            if first is None:
                self._rate_origin = {outer.key: (outer.completed, now)}
            else:
                done = outer.completed - first[0]
                hours = (now - first[1]) / 3600
                if done > 0 and hours > 0:
                    rate = Rate(unit="iter", per_hour=done / hours)
        parallel = None
        if self._region is not None and self.worker_region is None:
            region, total, ready = self._region
            seq = self.manager.last_committed_seq
            ready = ready or (seq is not None and seq != self._region_entry_seq)
            try:
                done = region.done_count()
            except Exception:
                done = 0
            parallel = ParallelStatus(region=region.region, total=total, done=done, ready=ready)
        return StatusSnapshot(
            progress=progress,
            rate=rate,
            last_checkpoint_seq=self.manager.last_committed_seq,
            message="replaying to resume point" if not self.reached_resume_point else "",
            parallel=parallel,
        )


def _is_iteration_of(position: Position, pending: dict[str, Any]) -> bool:
    """Is ``position`` the end of an iteration of the pending chunk's loop (not a nested one)?"""
    path = position.get("path") or []
    return (
        position.get("end") == "iter"
        and position.get("stmt") == pending["stmt"]
        and len(path) == pending["depth"] + 1
        and path[-1][0] == pending["event"]
    )


def _sampler_driven(iterable: Any) -> bool:
    """Iteration order drawn from the global RNG when iteration starts (torch DataLoader)."""
    return hasattr(iterable, "batch_sampler") or hasattr(iterable, "sampler")


def _skip_batches(it: Any, count: int) -> bool:
    """Advance a single-process, map-style DataLoader iterator by ``count`` batches without
    loading them (only the sampler moves). False if ``it`` is not such an iterator."""
    if type(it).__name__ != "_SingleProcessDataLoaderIter" or getattr(it, "_dataset_kind", None):
        return False  # _dataset_kind 0 = map-style
    if not callable(getattr(it, "_next_index", None)) or not hasattr(it, "_num_yielded"):
        return False
    for _ in builtins.range(count):
        it._next_index()
    it._num_yielded += count
    return True


def set_frame_locals(frame: types.FrameType, values: dict[str, Any]) -> None:
    """Write local variables of a running function (restore inside a function's loop).

    Python 3.13+ (PEP 667): ``frame.f_locals`` writes through. Older CPython: update the
    locals dict, then copy it into the fast locals.
    """
    if sys.version_info >= (3, 13):
        proxy = frame.f_locals
        for name, value in values.items():
            proxy[name] = value
        return
    import ctypes

    frame.f_locals.update(values)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
