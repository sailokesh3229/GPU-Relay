"""Where jobs on this computer run (plan P15 and the packaged desktop app).

* :func:`detect_gpu` names this computer's NVIDIA GPU (``nvidia-smi``), so the scheduler
  counts the local backend as a GPU platform with its real speed instead of a CPU.
* :func:`runtime_python` is the interpreter for local segments. From a normal install it is
  this Python, which has GPU Relay. The packaged app (PyInstaller) has no Python of its own,
  so on first use it makes a virtual environment from the Python installed on the computer
  and installs the GPU Relay runtime into it, exactly as cloud sessions do.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)

MARKER = "gpurelay-runtime.txt"  # in the environment: the runtime version installed there
_lock = threading.Lock()


class NoPythonError(RuntimeError):
    """The packaged app needs an installed Python to run jobs on this computer."""


def detect_gpu(run: object = subprocess.run) -> str | None:
    """The first NVIDIA GPU's name (e.g. ``"NVIDIA GeForce RTX 3060"``), or None."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = run(  # type: ignore[operator]
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    names = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return names[0] if result.returncode == 0 and names else None


def system_python() -> list[str] | None:
    """A command that starts an installed Python 3.10+ (not the Microsoft Store stub)."""
    candidates: list[list[str]] = []
    if sys.platform == "win32" and shutil.which("py"):
        candidates.append(["py", "-3"])
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found and "WindowsApps" not in found:  # the Store alias opens the Store instead
            candidates.append([found])
    for cmd in candidates:
        try:
            out = subprocess.run(
                [*cmd, "-c", "import sys; print(sys.version_info >= (3, 10))"],
                capture_output=True, text=True, timeout=30, check=False,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip() == "True":
            return cmd
    return None


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_python(home: Path) -> str:
    """The interpreter for local segments (see the module docstring). Raises
    :class:`NoPythonError` from the packaged app when no Python is installed."""
    if not frozen():
        return sys.executable
    import gpurelay
    from gpurelay.bundle.wheel import build_runtime_wheel

    venv = home / "local-runtime"
    python = _venv_python(venv)
    marker = venv / MARKER
    with _lock:
        if (
            python.exists()
            and marker.exists()
            and marker.read_text().strip() == (gpurelay.__version__)
        ):
            return str(python)
        base = system_python()
        if base is None:
            raise NoPythonError(
                "running jobs on this computer needs Python 3.10 or newer: install it from "
                "python.org (or use Kaggle and Colab, which need nothing here)"
            )
        log.info("preparing the local job environment in %s (first run only)", venv)
        if not python.exists():
            subprocess.run([*base, "-m", "venv", str(venv)], check=True, timeout=600)
        wheel = build_runtime_wheel(home / "local-runtime-wheel")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
             "--upgrade", str(wheel)],
            check=True, timeout=1800,
        )  # fmt: skip
        marker.write_text(gpurelay.__version__, encoding="utf-8")
        return str(python)
