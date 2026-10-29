"""Filesystem path helpers."""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath

_EXTENDED = "\\\\?\\"


def long_path(path: Path) -> Path:
    """Absolute path that also works beyond Windows' 260-character ``MAX_PATH``.

    On Windows this is the extended-length form (``\\\\?\\C:\\...``); elsewhere it is
    simply the resolved path. Content-addressed keys (64-hex digests) under deep
    session directories easily exceed ``MAX_PATH``.
    """
    resolved = Path(path).resolve()
    return _extended(resolved) if sys.platform == "win32" else resolved


def _extended(resolved: Path) -> Path:
    text = str(resolved)
    if text.startswith(_EXTENDED):
        return resolved
    if text.startswith("\\\\"):  # UNC share: \\server\share -> \\?\UNC\server\share
        return Path(_EXTENDED + "UNC\\" + text[2:])
    return Path(_EXTENDED + text)


class UnsafePathError(ValueError):
    """A stored relative path would escape its directory (absolute, drive, or ``..``)."""


_DRIVE = re.compile(r"^[A-Za-z]:")


def safe_join(root: Path, rel: str) -> Path:
    """``root / rel`` for a relative POSIX path read from storage, refusing any escape.

    Paths in bundles and checkpoints come from real files, but a damaged or tampered store
    must never make GPU Relay write outside the job's directories (e.g. ``../../.bashrc``).
    """
    text = rel.replace("\\", "/")
    parts = PurePosixPath(text).parts
    if (
        not text
        or text.startswith("/")
        or _DRIVE.match(text)
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise UnsafePathError(f"unsafe relative path {rel!r}")
    return Path(root).joinpath(*parts)
