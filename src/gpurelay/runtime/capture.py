"""Output capture (spec §9).

* ``sys.stdout`` / ``sys.stderr`` are replaced by tees that write to the session console
  and to one segment log file, counting **bytes written** (the offset recorded at every
  checkpoint).
* ``matplotlib.pyplot.show`` is patched to save every open figure as PNG into the blob
  cache (and close it), recording the figure order.
* While a resumed segment replays toward its resume point, recording is off and nothing
  is echoed, so replayed code does not duplicate output.
* Python ``warnings`` shown in earlier segments are not shown again (Python prints most
  warnings once per location *per process*; this extends that to the whole job).

Output written directly to OS file descriptors (C extensions, child processes) is not
captured in v1.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import sys
import threading
import types
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO

from gpurelay.runtime.blobs import BlobCache
from gpurelay.runtime.importhooks import ImportHook, when_imported

log = logging.getLogger(__name__)

LOG_ENCODING = "utf-8"


class _Tee(io.TextIOBase):
    """Text stream that forwards to the console and the capture log."""

    def __init__(self, capture: OutputCapture, console: TextIO, name: str) -> None:
        super().__init__()
        self._capture = capture
        self._console = console
        self.name = name

    def write(self, s: str) -> int:
        return self._capture._write(self._console, s, self.name)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        # The console may already be closed (e.g. at interpreter exit).
        with contextlib.suppress(ValueError, OSError):
            self._console.flush()
        self._capture.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return self._console.isatty()

    def fileno(self) -> int:
        return self._console.fileno()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._console, "encoding", None) or LOG_ENCODING

    @property
    def errors(self) -> str | None:  # type: ignore[override]
        return getattr(self._console, "errors", None)


class OutputCapture:
    """Tees stdout/stderr into ``log_path`` and captures matplotlib figures."""

    def __init__(self, log_path: Path, blobs: BlobCache, *, recording: bool = True) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = blobs
        self.recording = recording
        self._muted = 0
        self.offset = 0
        self.figures: list[str] = []
        self.warnings_shown: set[str] = set()  # this job, including earlier segments
        self._saved_showwarning: Any = None
        # Notebook executor hooks: per-cell stream outputs and figures.
        self.stream_listener: Callable[[str, str], None] | None = None
        self.figure_listener: Callable[[bytes], None] | None = None
        self._lock = threading.RLock()
        self._fh: io.BufferedWriter | None = None
        self._saved_streams: tuple[TextIO, TextIO] | None = None
        self._saved_show: Any = None
        self._hook: ImportHook | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._fh = open(self.log_path, "wb")  # noqa: SIM115 - closed in stop()
        self._saved_streams = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(self, sys.stdout, "stdout")
        sys.stderr = _Tee(self, sys.stderr, "stderr")
        self._install_figure_hook()
        self._install_warning_hook()

    def stop(self) -> None:
        if self._saved_streams is not None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # pragma: no cover - best effort
                pass
            sys.stdout, sys.stderr = self._saved_streams
            self._saved_streams = None
        self._uninstall_figure_hook()
        if self._saved_showwarning is not None:
            warnings.showwarning = self._saved_showwarning
            self._saved_showwarning = None
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    # ------------------------------------------------------------------ streams

    def _write(self, console: TextIO, s: str, stream: str = "stdout") -> int:
        with self._lock:
            if self.recording and not self._muted:
                console.write(s)
                if self._fh is not None:
                    data = s.encode(LOG_ENCODING, "backslashreplace")
                    self._fh.write(data)
                    self.offset += len(data)
                if self.stream_listener is not None and s:
                    self.stream_listener(stream, s)
        return len(s)

    @property
    def active(self) -> bool:
        """True when output is currently being recorded."""
        return self.recording and not self._muted

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    @contextlib.contextmanager
    def muted(self) -> Iterator[None]:
        """Drop all output (console and log) — for work the relay does on the user's behalf."""
        with self._lock:
            self._muted += 1
        try:
            yield
        finally:
            with self._lock:
                self._muted -= 1

    def set_recording(self, recording: bool) -> None:
        with self._lock:
            self.recording = recording

    def copy_log(self, dest: Path, upto: int) -> None:
        """Copy the first ``upto`` bytes of the log to ``dest``."""
        self.flush()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "rb") as src, open(dest, "wb") as out:
            out.write(src.read(upto))

    # ------------------------------------------------------------------ warnings

    def _install_warning_hook(self) -> None:
        original = warnings.showwarning
        self._saved_showwarning = original

        def showwarning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: TextIO | None = None,
            line: str | None = None,
        ) -> None:
            key = f"{category.__name__}:{os.path.basename(filename)}:{lineno}:{message}"
            if key in self.warnings_shown:
                return
            if self.recording:
                self.warnings_shown.add(key)
            original(message, category, filename, lineno, file, line)

        warnings.showwarning = showwarning

    # ------------------------------------------------------------------ figures

    def _install_figure_hook(self) -> None:
        """Patch ``pyplot.show`` now if loaded, else as soon as user code imports it."""
        if importlib.util.find_spec("matplotlib") is None:
            return
        os.environ.setdefault("MPLBACKEND", "Agg")
        self._hook = when_imported("matplotlib.pyplot", self._patch_pyplot)

    def _patch_pyplot(self, plt: types.ModuleType) -> None:
        self._saved_show = plt.show

        def show(*_args: Any, **_kwargs: Any) -> None:
            self.capture_open_figures()

        plt.show = show  # type: ignore[attr-defined]

    def _uninstall_figure_hook(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        plt = sys.modules.get("matplotlib.pyplot")
        if self._saved_show is not None and plt is not None:
            plt.show = self._saved_show  # type: ignore[attr-defined]
        self._saved_show = None

    def capture_open_figures(self) -> None:
        """Save every open matplotlib figure (in figure-number order) and close it."""
        import matplotlib.pyplot as plt

        for num in plt.get_fignums():
            fig = plt.figure(num)
            if self.active:
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                png = buf.getvalue()
                with self._lock:
                    self.figures.append(self.blobs.put_bytes(png))
                if self.figure_listener is not None:
                    self.figure_listener(png)
            plt.close(fig)
