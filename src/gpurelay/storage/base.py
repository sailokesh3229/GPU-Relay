"""Storage interface (spec §11.1) and remote-key validation.

Remote keys are POSIX-style relative paths such as ``jobs/job-1/status.json``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from gpurelay.constants import TMP_MARKER
from gpurelay.errors import InvalidRemotePathError

# Characters that are invalid in Windows file names (or would create NTFS alternate
# data streams). Rejected everywhere so keys stay portable across storage backends.
_FORBIDDEN_CHARS = frozenset('<>:"|?*\\')


@runtime_checkable
class Storage(Protocol):
    """Durable shared store reachable from every backend and the controller.

    Implementations must make ``put_file``/``put_bytes`` atomic from a reader's
    perspective: a concurrent reader sees either the old content or the new content,
    never a partial file.
    """

    def put_file(self, local: Path, remote: str) -> None:
        """Upload the local file ``local`` to ``remote``, replacing any existing key."""
        ...

    def get_file(self, remote: str, local: Path) -> None:
        """Download ``remote`` to ``local``. Raises ``StorageNotFoundError`` if missing."""
        ...

    def put_bytes(self, data: bytes, remote: str) -> None:
        """Store ``data`` at ``remote``, replacing any existing key."""
        ...

    def get_bytes(self, remote: str) -> bytes | None:
        """Return the content of ``remote``, or ``None`` if it does not exist."""
        ...

    def list(self, prefix: str) -> list[str]:
        """Return all keys starting with ``prefix`` (string prefix match), sorted."""
        ...

    def delete(self, remote: str) -> None:
        """Delete ``remote``. Deleting a missing key is not an error."""
        ...

    def exists(self, remote: str) -> bool:
        """Return True if ``remote`` exists."""
        ...


@runtime_checkable
class ChecksumStorage(Protocol):
    """Optional extension: storage that can report a stored object's SHA-256 cheaply.

    Used to verify uploads without downloading them again.
    """

    def sha256(self, remote: str) -> str | None:
        """SHA-256 hex digest of ``remote`` as stored, or None if missing."""
        ...


@runtime_checkable
class BatchStorage(Protocol):
    """Optional extension: many writes/deletes in one operation (one commit on the Hub)."""

    def put_many(self, items: Sequence[tuple[str, Path | bytes]], message: str = ...) -> None:
        """Store every (remote key, local file or bytes) pair."""
        ...

    def delete_many(self, remotes: Iterable[str], message: str = ...) -> None:
        """Delete every key; missing keys are ignored."""
        ...


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 hex digest of a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_key(remote: str) -> str:
    """Validate a remote key and return it unchanged.

    A valid key is a non-empty relative POSIX path with no empty, ``.`` or ``..``
    components, no Windows-forbidden characters, and no temp-file marker.
    """
    if not remote:
        raise InvalidRemotePathError("remote key must not be empty")
    if remote.startswith("/"):
        raise InvalidRemotePathError(f"remote key must be relative: {remote!r}")
    _check_chars(remote)
    for part in remote.split("/"):
        if part in ("", ".", ".."):
            raise InvalidRemotePathError(f"invalid path component in remote key: {remote!r}")
        if part != part.strip() or part.endswith("."):
            raise InvalidRemotePathError(f"non-portable path component in remote key: {remote!r}")
    return remote


def validate_prefix(prefix: str) -> str:
    """Validate a listing prefix. Empty means "everything"; a trailing ``/`` is allowed."""
    if not prefix:
        return prefix
    if prefix.startswith("/"):
        raise InvalidRemotePathError(f"prefix must be relative: {prefix!r}")
    _check_chars(prefix)
    parts = prefix.split("/")
    # Last part may be empty (trailing slash) or a partial name.
    for part in parts[:-1]:
        if part in ("", ".", ".."):
            raise InvalidRemotePathError(f"invalid path component in prefix: {prefix!r}")
    return prefix  # the partial last part is only string-matched, never used as a path


def _check_chars(value: str) -> None:
    if TMP_MARKER in value:
        raise InvalidRemotePathError(f"remote key may not contain {TMP_MARKER!r}: {value!r}")
    bad = {c for c in value if c in _FORBIDDEN_CHARS or ord(c) < 32}
    if bad:
        raise InvalidRemotePathError(f"forbidden characters {sorted(bad)!r} in {value!r}")
