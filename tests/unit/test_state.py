from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

import pytest

from gpurelay.runtime.state import capture_state, restore_state, skip_reason


class Point:
    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and other.x == self.x


def _gen() -> Any:
    yield 1


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("__relay__", object(), "gpurelay runtime"),
        ("__name__", "x", "dunder name"),
        ("In", [], "IPython internal"),
        ("_i3", "", "IPython internal"),
        ("_12", 1, "IPython internal"),
        ("math", math, "module"),
        ("Point", Point, "class"),
        ("helper", lambda: 1, "function"),
        ("sqrt", math.sqrt, "function"),
    ],
)
def test_skip_reasons(name: str, value: Any, reason: str) -> None:
    assert skip_reason(name, value) == reason


@pytest.mark.parametrize("name", ["losses", "_private", "epoch"])
def test_regular_names_not_skipped(name: str) -> None:
    assert skip_reason(name, 1) is None


def test_plain_round_trip_keeps_shared_references(tmp_path: Path) -> None:
    shared = [1, 2, 3]
    ns: dict[str, Any] = {
        "losses": [0.5, 0.25],
        "a": shared,
        "b": shared,
        "point": Point(4),
        "config": {"lr": 1e-3},
        "json": json,
        "fn": lambda x: x,
    }
    manifest = capture_state(ns, tmp_path)
    assert manifest.plain == ["a", "b", "config", "losses", "point"]
    assert manifest.skipped == {"fn": "function", "json": "module"}
    assert manifest.torch == []

    fresh: dict[str, Any] = {"losses": [], "unrelated": 1}
    report = restore_state(fresh, tmp_path)
    assert report.plain == ["a", "b", "config", "losses", "point"]
    assert fresh["losses"] == [0.5, 0.25]
    assert fresh["point"] == Point(4)
    assert fresh["a"] is fresh["b"]
    assert fresh["unrelated"] == 1


def test_selected_names_only(tmp_path: Path) -> None:
    ns = {"keep": 1, "drop": 2}
    manifest = capture_state(ns, tmp_path, names=["keep", "missing"])
    assert manifest.plain == ["keep"]
    fresh: dict[str, Any] = {}
    restore_state(fresh, tmp_path)
    assert fresh == {"keep": 1}


def test_unpicklable_values_reported_and_skipped(tmp_path: Path) -> None:
    fh = open(tmp_path / "f.txt", "w")  # noqa: SIM115
    try:
        ns = {"ok": [1], "fh": fh, "lock": threading.Lock(), "gen": _gen()}
        manifest = capture_state(ns, tmp_path)
    finally:
        fh.close()
    assert manifest.plain == ["ok"]
    assert set(manifest.unpicklable) == {"fh", "lock", "gen"}
    fresh: dict[str, Any] = {}
    restore_state(fresh, tmp_path)
    assert fresh == {"ok": [1]}


def test_empty_namespace_writes_nothing(tmp_path: Path) -> None:
    manifest = capture_state({}, tmp_path)
    assert manifest.plain == [] and manifest.torch == []
    assert list(tmp_path.iterdir()) == []


def test_import_bindings_are_recorded_and_imported_again(tmp_path: Path) -> None:
    import collections
    import json as js
    import os.path
    import types

    def own_helper() -> None: ...

    namespace = {
        "__name__": "__main__",
        "js": js,
        "OD": collections.OrderedDict,
        "pjoin": os.path.join,
        "own_helper": own_helper,  # defined by the job: recreated by replay, not imported
        "fake": types.ModuleType("not_importable"),
        "x": 1,
    }
    own_helper.__module__ = "__main__"
    manifest = capture_state(namespace, tmp_path)
    assert manifest.imports == {
        "js": ["json"],
        "OD": ["collections", "OrderedDict"],
        "pjoin": [os.path.join.__module__, "join"],
    }
    restored: dict[str, Any] = {"js": "replayed value wins"}
    report = restore_state(restored, tmp_path)
    assert restored["OD"] is collections.OrderedDict and restored["pjoin"] is os.path.join
    assert restored["js"] == "replayed value wins"
    assert report.imported == ["OD", "pjoin"] and restored["x"] == 1
    assert "own_helper" not in restored and "fake" not in restored


def test_rng_objects_are_restored_into_the_live_object() -> None:
    import numpy as np

    from gpurelay.runtime.session import apply_in_place, is_rng_object

    for make, draw in [
        (lambda: __import__("random").Random(1), lambda g: g.random()),
        (lambda: np.random.default_rng(1), lambda g: float(g.normal())),
        (lambda: np.random.RandomState(1), lambda g: float(g.uniform())),
    ]:
        saved = make()
        draw(saved)
        draw(saved)
        live = make()  # rebuilt by replay, still referenced elsewhere
        assert is_rng_object(live)
        kept = apply_in_place(live, saved)
        assert kept is live and draw(live) == draw(saved)
    assert not is_rng_object([1]) and not is_rng_object(np.zeros(2))
