"""Local content-addressed cache mirroring ``jobs/<job>/blobs/<sha256>`` in storage.

Files in the cache are immutable (named by their SHA-256), so the persist thread can
upload them while user code keeps running.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

from gpurelay.errors import ChecksumMismatchError
from gpurelay.pathutil import long_path
from gpurelay.storage.base import Storage, sha256_file
from gpurelay.storage.layout import blob_key


class BlobCache:
    """Local blob directory with lazy download from storage."""

    def __init__(self, root: Path, storage: Storage, job_id: str) -> None:
        self.root = long_path(Path(root))
        self.root.mkdir(parents=True, exist_ok=True)
        self.storage = storage
        self.job_id = job_id

    def path(self, sha: str) -> Path:
        return self.root / sha

    def _tmp(self) -> Path:
        return self.root / f".tmp-{uuid.uuid4().hex[:12]}"

    def _adopt(self, tmp: Path, sha: str) -> str:
        dest = self.path(sha)
        if dest.exists():
            tmp.unlink(missing_ok=True)
        else:
            os.replace(tmp, dest)
        return sha

    def put_bytes(self, data: bytes) -> str:
        """Store ``data``; return its SHA-256."""
        sha = hashlib.sha256(data).hexdigest()
        if not self.path(sha).exists():
            tmp = self._tmp()
            tmp.write_bytes(data)
            self._adopt(tmp, sha)
        return sha

    def put_file(self, src: Path) -> str:
        """Copy ``src`` into the cache (hashing while copying); return its SHA-256."""
        tmp = self._tmp()
        digest = hashlib.sha256()
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while chunk := fin.read(1 << 20):
                digest.update(chunk)
                fout.write(chunk)
        return self._adopt(tmp, digest.hexdigest())

    def fetch(self, sha: str) -> Path:
        """Return the local path of blob ``sha``, downloading and verifying if needed."""
        dest = self.path(sha)
        if dest.exists():
            return dest
        tmp = self._tmp()
        self.storage.get_file(blob_key(self.job_id, sha), tmp)
        actual = sha256_file(tmp)
        if actual != sha:
            tmp.unlink(missing_ok=True)
            raise ChecksumMismatchError(f"blob {sha}: downloaded content hashes to {actual}")
        self._adopt(tmp, sha)
        return dest

    def read_bytes(self, sha: str) -> bytes:
        return self.fetch(sha).read_bytes()

    def copy_to(self, sha: str, dest: Path) -> None:
        """Materialize blob ``sha`` at ``dest`` (parents created)."""
        src = self.fetch(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
