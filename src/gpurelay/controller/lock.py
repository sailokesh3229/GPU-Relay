"""One controller per GPU Relay home: an OS file lock held while ``gpurelay up`` runs.

The OS releases the lock when the process exits (even if killed), so it never goes stale.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from gpurelay.errors import ControllerRunningError

LOCK_FILENAME = "controller.lock"
_LOCK_OFFSET = 1024  # lock a byte past the pid text so other processes can still read it


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def controller_lock(home: Path) -> Iterator[None]:
    """Hold the controller lock for ``home``; raise ControllerRunningError if it is taken."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / LOCK_FILENAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if not _try_lock(fd):
            holder = path.read_text(encoding="utf-8", errors="replace").strip() or "unknown"
            raise ControllerRunningError(
                f"another controller is already running (pid {holder}); stop it first"
            )
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def controller_running(home: Path) -> bool:
    """True if a ``gpurelay up`` currently holds the lock for ``home`` (non-blocking probe)."""
    path = home / LOCK_FILENAME
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        if not _try_lock(fd):
            return True
        _unlock(fd)
        return False
    finally:
        os.close(fd)
