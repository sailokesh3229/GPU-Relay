"""Local filesystem storage, used for tests and local-only runs (spec §11.2).

Writes go to a temporary sibling file which is fsynced and then renamed over the
destination with :func:`os.replace`, so readers never observe a partial file.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import BinaryIO, TypeVar

from gpurelay.constants import TMP_MARKER
from gpurelay.errors import StorageNotFoundError
from gpurelay.pathutil import long_path
from gpurelay.storage.base import sha256_file, validate_key, validate_prefix

_T = TypeVar("_T")

# On Windows, renaming over a file that another thread/process has open, or opening a
# file that is mid-rename, fails transiently with PermissionError. Retry briefly.
_RETRY_ATTEMPTS = 200 if sys.platform == "win32" else 1
_RETRY_DELAY_S = 0.005


def _retry_os(fn: Callable[[], _T]) -> _T:
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except PermissionError:
            if attempt == _RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_DELAY_S)
    raise AssertionError("unreachable")


class LocalFSStorage:
    """:class:`~gpurelay.storage.base.Storage` backed by a local directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._root = long_path(self.root)  # used for all I/O (Windows MAX_PATH)
        self._root.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"LocalFSStorage({str(self.root)!r})"

    # ------------------------------------------------------------------ helpers

    def _path(self, remote: str) -> Path:
        return self._root.joinpath(*validate_key(remote).split("/"))

    def _atomic_write(self, dest: Path, write: Callable[[BinaryIO], object]) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}{TMP_MARKER}{uuid.uuid4().hex[:12]}")
        try:
            with open(tmp, "wb") as fh:
                write(fh)
                fh.flush()
                os.fsync(fh.fileno())
            _retry_os(lambda: os.replace(tmp, dest))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ Storage API

    def put_file(self, local: Path, remote: str) -> None:
        """Copy ``local`` into the store at ``remote`` atomically."""
        dest = self._path(remote)
        with open(local, "rb") as src:
            self._atomic_write(dest, lambda fh: shutil.copyfileobj(src, fh))

    def get_file(self, remote: str, local: Path) -> None:
        """Copy ``remote`` to ``local`` (streamed; atomic on the local side)."""
        path = self._path(remote)
        try:
            src = _retry_os(lambda: open(path, "rb"))  # noqa: SIM115 - closed below
        except (FileNotFoundError, IsADirectoryError):
            raise StorageNotFoundError(remote) from None
        except PermissionError:
            if path.is_dir():
                raise StorageNotFoundError(remote) from None
            raise
        with src:
            self._atomic_write(Path(local), lambda fh: shutil.copyfileobj(src, fh))

    def put_bytes(self, data: bytes, remote: str) -> None:
        """Store ``data`` at ``remote`` atomically."""
        dest = self._path(remote)
        self._atomic_write(dest, lambda fh: fh.write(data))

    def get_bytes(self, remote: str) -> bytes | None:
        """Return the content of ``remote``, or ``None`` if missing."""
        path = self._path(remote)
        try:
            return _retry_os(path.read_bytes)
        except (FileNotFoundError, IsADirectoryError):
            return None
        except PermissionError:
            # On Windows reading a directory raises PermissionError.
            if path.is_dir():
                return None
            raise

    def list(self, prefix: str) -> list[str]:
        """Return all keys starting with ``prefix``, sorted. Temp files are hidden."""
        validate_prefix(prefix)
        # Walk only the deepest directory fully named by the prefix.
        dir_part = prefix.rpartition("/")[0]
        base = self._root.joinpath(*dir_part.split("/")) if dir_part else self._root
        if not base.is_dir():
            return []
        keys: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(self._root).as_posix()
            for name in filenames:
                if TMP_MARKER in name:
                    continue
                key = name if rel_dir == "." else f"{rel_dir}/{name}"
                if key.startswith(prefix):
                    keys.append(key)
        keys.sort()
        return keys

    def delete(self, remote: str) -> None:
        """Delete ``remote``; missing keys are ignored."""
        path = self._path(remote)
        if path.is_dir():
            return  # a directory is not a key (only a prefix of other keys): nothing to delete
        _retry_os(lambda: path.unlink(missing_ok=True))

    def exists(self, remote: str) -> bool:
        """Return True if ``remote`` is an existing key."""
        return self._path(remote).is_file()

    def put_many(
        self, items: Sequence[tuple[str, Path | bytes]], message: str = "gpurelay batch"
    ) -> None:
        """Store several keys (each write is atomic; the batch as a whole is not)."""
        for remote, source in items:
            if isinstance(source, bytes):
                self.put_bytes(source, remote)
            else:
                self.put_file(source, remote)

    def delete_many(self, remotes: Iterable[str], message: str = "gpurelay delete") -> None:
        for remote in remotes:
            self.delete(remote)

    def sha256(self, remote: str) -> str | None:
        """SHA-256 hex digest of ``remote`` as stored, or None if missing."""
        path = self._path(remote)
        if not path.is_file():
            return None
        return _retry_os(lambda: sha256_file(path))
