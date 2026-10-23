"""Working-directory file tracking (spec §7.5 "Working files").

Files written by user code under the job working directory are tracked by
(mtime, size) and hashed when changed. A checkpoint records a manifest
``{relative_path: sha256}`` of every file that differs from the bundle; the bytes go to
the content-addressed blob store. On resume the manifest is materialized before replay,
and files that replay modified are reverted when the resume point is reached.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from gpurelay.runtime.blobs import BlobCache

log = logging.getLogger(__name__)

EXCLUDED_DIRS = frozenset({"__pycache__", ".ipynb_checkpoints", ".git"})


@dataclass(frozen=True)
class _Seen:
    mtime_ns: int
    size: int
    sha: str


class WorkFileTracker:
    """Tracks files under ``root`` relative to a baseline (the bundle's files)."""

    def __init__(
        self,
        root: Path,
        blobs: BlobCache,
        baseline: dict[str, str] | None = None,
        exclude: set[str] | None = None,
    ):
        self.root = Path(root)
        self.blobs = blobs
        self.baseline = dict(baseline or {})
        self.exclude = set(exclude or ())  # staged data: relative paths (files or dirs)
        self._seen: dict[str, _Seen] = {}

    def _walk(self) -> dict[str, os.stat_result]:
        found: dict[str, os.stat_result] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            if rel_dir != "." and self._excluded(rel_dir):
                dirnames[:] = []
                continue
            for name in filenames:
                path = Path(dirpath) / name
                rel = path.relative_to(self.root).as_posix()
                if self._excluded(rel):
                    continue
                try:
                    found[rel] = path.stat()
                except FileNotFoundError:
                    continue
        return found

    def _excluded(self, rel: str) -> bool:
        return any(rel == ex or rel.startswith(ex + "/") for ex in self.exclude)

    def scan(self) -> dict[str, str]:
        """Return the manifest of files that differ from the baseline.

        Changed files are copied into the blob cache (so later edits by user code cannot
        affect the checkpoint being persisted).
        """
        manifest: dict[str, str] = {}
        current: dict[str, _Seen] = {}
        for rel, st in sorted(self._walk().items()):
            seen = self._seen.get(rel)
            if seen is not None and seen.mtime_ns == st.st_mtime_ns and seen.size == st.st_size:
                sha = seen.sha
            else:
                try:
                    sha = self.blobs.put_file(self.root / rel)
                except FileNotFoundError:
                    continue
            current[rel] = _Seen(st.st_mtime_ns, st.st_size, sha)
            if self.baseline.get(rel) != sha:
                manifest[rel] = sha
        self._seen = current
        return manifest

    def materialize(self, manifest: dict[str, str]) -> None:
        """Write every manifest file into the working directory (overwriting)."""
        for rel, sha in sorted(manifest.items()):
            self.blobs.copy_to(sha, self.root / rel)
        self._remember(manifest)

    def revert(self, manifest: dict[str, str]) -> list[str]:
        """Restore manifest files whose content changed since :meth:`materialize`.

        Returns the reverted paths.
        """
        reverted: list[str] = []
        for rel, sha in sorted(manifest.items()):
            path = self.root / rel
            seen = self._seen.get(rel)
            try:
                st = path.stat()
                unchanged = (
                    seen is not None
                    and seen.sha == sha
                    and seen.mtime_ns == st.st_mtime_ns
                    and seen.size == st.st_size
                )
            except FileNotFoundError:
                unchanged = False
            if not unchanged:
                self.blobs.copy_to(sha, path)
                reverted.append(rel)
        self._remember(manifest)
        if reverted:
            log.info("reverted %d working file(s) modified during replay", len(reverted))
        return reverted

    def _remember(self, manifest: dict[str, str]) -> None:
        for rel, sha in manifest.items():
            st = (self.root / rel).stat()
            self._seen[rel] = _Seen(st.st_mtime_ns, st.st_size, sha)
