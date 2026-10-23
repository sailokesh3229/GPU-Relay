"""Property-based tests (hardening campaign): random inputs against models of the truth."""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Any

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from gpurelay.durations import format_duration, parse_duration  # noqa: E402
from gpurelay.errors import InvalidRemotePathError  # noqa: E402
from gpurelay.runtime.loops import Position, Tracker, run_loop  # noqa: E402
from gpurelay.runtime.state import capture_state, restore_state  # noqa: E402
from gpurelay.storage.base import validate_key  # noqa: E402
from gpurelay.storage.layout import ckpt_prefix, parse_ckpt_seq  # noqa: E402
from gpurelay.storage.localfs import LocalFSStorage  # noqa: E402

# GPURELAY_PROPERTY_EXAMPLES=600 for a deep run.
FAST = settings(max_examples=int(os.environ.get("GPURELAY_PROPERTY_EXAMPLES", "60")), deadline=None,
                suppress_health_check=[HealthCheck.function_scoped_fixture])  # fmt: skip

# ---------------------------------------------------------------- loop tracker

# A random program: a list of loops, each with a length and an optional nested loop length.
programs = st.lists(
    st.tuples(st.integers(0, 4), st.one_of(st.none(), st.integers(0, 3)), st.booleans()),
    min_size=1,
    max_size=4,
)


def _run(spec: list[tuple[int, int | None, bool]], tracker: Tracker, out: list[Any],
         on_sp: Any) -> None:  # fmt: skip
    def loop(n: int) -> Any:
        return run_loop(tracker, n, None, lambda s: ((i, i) for i in range(s, n)), on_sp)

    for k, (n, inner, explicit) in enumerate(spec):
        for a in loop(n):
            out.append((k, a))
            if inner is not None:
                for b in loop(inner):
                    out.append((k, a, b))
            if explicit:
                position = tracker.safe_point()
                if position is not None:
                    on_sp(position)
    tracker.end_statement()
    out.append("end")


@FAST
@given(programs)
def test_resume_from_every_safe_point_reproduces_the_rest(
    spec: list[tuple[int, int | None, bool]],
) -> None:
    out: list[Any] = []
    points: list[tuple[Position, int]] = []
    live = Tracker(on_reach=lambda: None)
    live.begin_statement(0)
    _run(spec, live, out, lambda pos: points.append((pos, len(out))))
    for position, done in points:
        resumed: list[Any] = []
        reached = []
        tracker = Tracker(on_reach=lambda flags=reached: flags.append(True))
        tracker.begin_statement(0, target=position)
        # Output while replaying (before the resume point) is suppressed by the runtime:
        # model that by recording only after the resume point is reached.
        _run(spec, tracker, resumed, lambda _pos: None)
        assert reached, position
        assert resumed[-(len(out) - done) :] == out[done:] if len(out) > done else True


# ---------------------------------------------------------------- storage

keys = st.lists(
    st.text(alphabet=string.ascii_lowercase + string.digits + "-_.", min_size=1, max_size=6).filter(
        lambda part: part not in (".", "..") and not part.endswith(".")
    ),
    min_size=1,
    max_size=3,
).map("/".join)


@FAST
@given(st.lists(st.tuples(st.sampled_from(["put", "delete"]), keys, st.binary(max_size=20)),
                max_size=25))  # fmt: skip
def test_localfs_matches_a_dict_model(tmp_path_factory: Any, ops: list[Any]) -> None:
    store = LocalFSStorage(tmp_path_factory.mktemp("store"))
    model: dict[str, bytes] = {}
    for op, key, data in ops:
        # A key cannot be both a file and a directory prefix of another key.
        if op == "put" and any(k.startswith(key + "/") or key.startswith(k + "/") for k in model):
            continue
        if op == "put":
            store.put_bytes(data, key)
            model[key] = data
        else:
            store.delete(key)
            model.pop(key, None)
    assert store.list("") == sorted(model)
    for key, data in model.items():
        assert store.get_bytes(key) == data and store.exists(key)
    for prefix in {k[:n] for k in model for n in range(len(k) + 1)}:
        assert store.list(prefix) == sorted(k for k in model if k.startswith(prefix))


@FAST
@given(st.text(max_size=12))
def test_key_validation_never_crashes_and_rejects_escapes(text: str) -> None:
    try:
        validate_key(text)
    except InvalidRemotePathError:
        return
    assert text and not text.startswith("/")
    assert all(part not in ("", ".", "..") for part in text.split("/"))


# ---------------------------------------------------------------- state

values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=8)
    | st.binary(max_size=8),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=4), children, max_size=4)
        | st.tuples(children, children)
    ),
    max_leaves=12,
)


@FAST
@given(st.dictionaries(st.from_regex(r"[a-z][a-z0-9_]{0,6}", fullmatch=True), values, max_size=6))
def test_state_round_trip(tmp_path_factory: Any, namespace: dict[str, Any]) -> None:
    out = tmp_path_factory.mktemp("state")
    capture_state(namespace, out)
    restored: dict[str, Any] = {}
    restore_state(restored, out)
    assert restored == namespace


# ---------------------------------------------------------------- small helpers


@FAST
@given(st.floats(min_value=0.001, max_value=1e6))
def test_duration_format_parses_back_closely(seconds: float) -> None:
    text = format_duration(seconds)
    assume(not text.startswith("0.0"))
    assert parse_duration(text) == pytest.approx(seconds, rel=0.06, abs=0.06)


@FAST
@given(st.text(alphabet=string.ascii_lowercase + string.digits + "-", min_size=1, max_size=12),
       st.integers(0, 10**9))  # fmt: skip
def test_checkpoint_layout_round_trip(job: str, seq: int) -> None:
    assert parse_ckpt_seq(ckpt_prefix(job, seq) + "/meta.json", job) == seq


def test_property_suite_imports_cleanly(tmp_path: Path) -> None:
    assert LocalFSStorage(tmp_path).list("") == []


@FAST
@given(st.lists(st.tuples(st.sampled_from(["put", "delete"]), keys, st.binary(max_size=20)),
                max_size=25))  # fmt: skip
def test_gdrive_matches_a_dict_model(ops: list[Any]) -> None:
    from test_gdrive_storage import FakeDrive

    from gpurelay.retry import RetryPolicy
    from gpurelay.storage.gdrive import FOLDER_MIME, GoogleDriveStorage

    drive = FakeDrive()
    drive.entries["root"] = {"id": "root", "name": "store", "mimeType": FOLDER_MIME,
                             "parent": "-", "data": None, "modifiedTime": "0"}  # fmt: skip
    store = GoogleDriveStorage("root", drive, retry=RetryPolicy(attempts=1, base_delay_s=0))
    model: dict[str, bytes] = {}
    for op, key, data in ops:
        if op == "put" and any(k.startswith(key + "/") or key.startswith(k + "/") for k in model):
            continue
        if op == "put":
            store.put_bytes(data, key)
            model[key] = data
        else:
            store.delete(key)
            model.pop(key, None)
    assert store.list("") == sorted(model)
    for key, data in model.items():
        assert store.get_bytes(key) == data
    fresh = GoogleDriveStorage("root", drive)  # no cache: same view
    for prefix in {k[:n] for k in model for n in range(len(k) + 1)}:
        assert fresh.list(prefix) == sorted(k for k in model if k.startswith(prefix))
