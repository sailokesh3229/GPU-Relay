"""Filesystem path helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_EXTENDED = "\\\\?\\"


def long_path(path: Path) -> Path:
    """Absolute path that also works beyond Windows' 260-character ``MAX_PATH``.

    On Windows this is the extended-length form (``\\\\?\\C:\\...``); elsewhere it is
    simply the resolved path. Content-addressed keys (64-hex digests) under deep
    session directories easily exceed ``MAX_PATH``.
    """
    resolved = Path(path).resolve()
    if sys.platform != "win32":
        return resolved
    text = str(resolved)
    if text.startswith(_EXTENDED):
        return resolved
    if text.startswith("\\\\"):  # UNC share: \\server\share -> \\?\UNC\server\share
        return Path(_EXTENDED + "UNC\\" + text[2:])
    return Path(_EXTENDED + text)
