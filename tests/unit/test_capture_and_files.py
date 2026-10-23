from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from gpurelay.durations import parse_duration
from gpurelay.runtime.blobs import BlobCache
from gpurelay.runtime.capture import OutputCapture
from gpurelay.runtime.workfiles import WorkFileTracker
from gpurelay.storage import LocalFSStorage


@pytest.fixture
def blobs(tmp_path: Path) -> BlobCache:
    return BlobCache(tmp_path / "blobs", LocalFSStorage(tmp_path / "store"), "job")


# ---------------------------------------------------------------- capture


def test_tee_counts_bytes_and_respects_recording(
    tmp_path: Path, blobs: BlobCache, capsys: pytest.CaptureFixture[str]
) -> None:
    cap = OutputCapture(tmp_path / "log.txt", blobs, recording=False)
    cap.start()
    try:
        print("replayed - must not appear")
        cap.set_recording(True)
        print("héllo")
        print("err line", file=sys.stderr)
    finally:
        cap.stop()
    data = (tmp_path / "log.txt").read_bytes()
    assert data == "héllo\nerr line\n".encode()
    assert cap.offset == len(data)
    out = capsys.readouterr()
    assert "replayed" not in out.out and "héllo" in out.out


def test_copy_log_prefix(tmp_path: Path, blobs: BlobCache) -> None:
    cap = OutputCapture(tmp_path / "log.txt", blobs)
    cap.start()
    try:
        print("one")
        offset = cap.offset
        print("two")
        cap.copy_log(tmp_path / "copy.txt", offset)
    finally:
        cap.stop()
    assert (tmp_path / "copy.txt").read_bytes() == b"one\n"


def test_show_saves_figures(tmp_path: Path, blobs: BlobCache) -> None:
    pytest.importorskip("matplotlib")
    cap = OutputCapture(tmp_path / "log.txt", blobs)
    cap.start()
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot([1, 2, 3])
        plt.figure()
        plt.plot([3, 2, 1])
        plt.show()
        assert plt.get_fignums() == []
    finally:
        cap.stop()
    assert len(cap.figures) == 2
    assert blobs.path(cap.figures[0]).read_bytes().startswith(b"\x89PNG")


# ---------------------------------------------------------------- working files


def test_manifest_tracks_changes_against_baseline(tmp_path: Path, blobs: BlobCache) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "script.py").write_text("print(1)")
    baseline = {"script.py": blobs.put_file(work / "script.py")}
    tracker = WorkFileTracker(work, blobs, baseline)
    assert tracker.scan() == {}

    (work / "out").mkdir()
    (work / "out" / "model.bin").write_bytes(b"weights-1")
    (work / "__pycache__").mkdir()
    (work / "__pycache__" / "x.pyc").write_bytes(b"ignored")
    manifest = tracker.scan()
    assert set(manifest) == {"out/model.bin"}
    assert blobs.path(manifest["out/model.bin"]).read_bytes() == b"weights-1"

    (work / "out" / "model.bin").write_bytes(b"weights-22")
    assert blobs.path(tracker.scan()["out/model.bin"]).read_bytes() == b"weights-22"
    (work / "out" / "model.bin").unlink()
    assert tracker.scan() == {}


def test_materialize_and_revert(tmp_path: Path, blobs: BlobCache) -> None:
    work = tmp_path / "work"
    work.mkdir()
    manifest = {"log.csv": blobs.put_bytes(b"a,b\n1,2\n")}
    tracker = WorkFileTracker(work, blobs)
    tracker.materialize(manifest)
    assert (work / "log.csv").read_bytes() == b"a,b\n1,2\n"

    # Replayed code truncates the file ...
    (work / "log.csv").write_bytes(b"a,b\n")
    os.utime(work / "log.csv", ns=(1, 1))
    # ... reaching the resume point restores the checkpointed version.
    assert tracker.revert(manifest) == ["log.csv"]
    assert (work / "log.csv").read_bytes() == b"a,b\n1,2\n"
    assert tracker.revert(manifest) == []


# ---------------------------------------------------------------- durations


@pytest.mark.parametrize(
    ("text", "seconds"), [("90", 90), ("20s", 20), ("5m", 300), ("1.5h", 5400), ("2d", 172800)]
)
def test_parse_duration(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "5x", "-3s", "0"])
def test_parse_duration_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(text)


def test_parse_duration_zero_when_allowed() -> None:
    assert parse_duration("0h", allow_zero=True) == 0.0
    with pytest.raises(ValueError):
        parse_duration("-1h", allow_zero=True)
