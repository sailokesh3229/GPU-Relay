"""Hugging Face Hub storage (spec §11.2): one private dataset repo, e.g. ``<user>/gpurelay-store``.

Every write is a git commit on the Hub, so this storage also implements
:class:`~gpurelay.storage.base.BatchStorage` (many files per commit; the checkpoint persister
uses it) and keeps single-file writes for small JSON documents. Reads bypass the local Hub
cache (``status.json`` / ``LATEST.json`` must always be fresh).

Errors are mapped onto the storage exceptions: missing entries -> ``StorageNotFoundError``
(not retried), rate limits / server / network errors -> ``StorageError`` (retried).

VERIFY: commit rate limits and private-repo storage limits on the Hub (docs/verify.md #9).
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from gpurelay.errors import StorageError, StorageNotFoundError
from gpurelay.retry import RetryPolicy
from gpurelay.storage.base import sha256_file, validate_key, validate_prefix

log = logging.getLogger(__name__)

_T = TypeVar("_T")
REPO_TYPE = "dataset"
_RETRY = RetryPolicy(attempts=6, base_delay_s=2.0, max_delay_s=60.0)


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    return (
        name
        in {
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
            "LocalEntryNotFoundError",
            "RepositoryNotFoundError",
            "RevisionNotFoundError",
        }
        or _status_code(exc) == 404
    )


def _quiet_hub() -> None:
    """No progress bars or per-request logs: inside a session they would land in the user's
    captured stderr (and break log equivalence); in the controller they are just noise."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars  # type: ignore[attr-defined]

        disable_progress_bars()
    except ImportError:  # pragma: no cover - very old huggingface_hub
        pass
    for name in ("httpx", "httpx2", "httpcore", "huggingface_hub", "hf_xet"):
        logging.getLogger(name).setLevel(logging.WARNING)


class HFHubStorage:
    """:class:`~gpurelay.storage.base.Storage` on a private Hugging Face dataset repo."""

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        *,
        api: Any = None,
        downloader: Callable[..., str] | None = None,
        retry: RetryPolicy = _RETRY,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        _quiet_hub()
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.api = api
        if downloader is None:
            from huggingface_hub import hf_hub_download

            downloader = hf_hub_download
        self._download = downloader
        self.retry = retry

    def __repr__(self) -> str:
        return f"HFHubStorage({self.repo_id!r})"

    # ------------------------------------------------------------------ plumbing

    def _call(self, fn: Callable[[], _T], what: str) -> _T:
        def attempt() -> _T:
            try:
                return fn()
            except (StorageError, FileNotFoundError):
                raise
            except Exception as exc:
                if _is_not_found(exc):
                    raise StorageNotFoundError(f"{what}: not found") from exc
                raise StorageError(f"{what}: {type(exc).__name__}: {exc}") from exc

        return self.retry.call(attempt, what=f"hf {what}")

    def _commit(self, operations: list[Any], message: str) -> None:
        if not operations:
            return
        self._call(
            lambda: self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=REPO_TYPE,
                operations=operations,
                commit_message=message,
                token=self.token,
            ),
            f"commit ({message})",
        )

    @staticmethod
    def _add(remote: str, source: Path | bytes) -> Any:
        from huggingface_hub import CommitOperationAdd

        payload = source if isinstance(source, bytes) else str(source)
        return CommitOperationAdd(path_in_repo=validate_key(remote), path_or_fileobj=payload)

    @staticmethod
    def _delete(remote: str) -> Any:
        from huggingface_hub import CommitOperationDelete

        return CommitOperationDelete(path_in_repo=validate_key(remote))

    # ------------------------------------------------------------------ Storage API

    def put_file(self, local: Path, remote: str) -> None:
        self._commit([self._add(remote, Path(local))], f"put {remote}")

    def put_bytes(self, data: bytes, remote: str) -> None:
        self._commit([self._add(remote, data)], f"put {remote}")

    def get_file(self, remote: str, local: Path) -> None:
        validate_key(remote)
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gpurelay-dl-", dir=local.parent) as tmp:
            downloaded = self._call(
                lambda: self._download(
                    repo_id=self.repo_id,
                    filename=remote,
                    repo_type=REPO_TYPE,
                    token=self.token,
                    local_dir=tmp,
                    force_download=True,
                ),
                f"get {remote}",
            )
            os.replace(downloaded, local)

    def get_bytes(self, remote: str) -> bytes | None:
        with tempfile.TemporaryDirectory(prefix="gpurelay-dl-") as tmp:
            target = Path(tmp) / "data"
            try:
                self.get_file(remote, target)
            except StorageNotFoundError:
                return None
            return target.read_bytes()

    def list(self, prefix: str) -> list[str]:
        validate_prefix(prefix)
        directory = prefix.rpartition("/")[0] or None

        def entries() -> list[str]:
            items = self.api.list_repo_tree(
                self.repo_id,
                path_in_repo=directory,
                recursive=True,
                repo_type=REPO_TYPE,
                token=self.token,
            )
            # RepoFile has a size; RepoFolder does not.
            return [i.path for i in items if hasattr(i, "size") and not hasattr(i, "tree_id")]

        try:
            paths = self._call(entries, f"list {prefix or '/'}")
        except StorageNotFoundError:
            return []
        return sorted(p for p in paths if p.startswith(prefix))

    def delete(self, remote: str) -> None:
        self.delete_many([remote])

    def exists(self, remote: str) -> bool:
        validate_key(remote)
        return bool(
            self._call(
                lambda: self.api.file_exists(
                    self.repo_id, remote, repo_type=REPO_TYPE, token=self.token
                ),
                f"exists {remote}",
            )
        )

    # ------------------------------------------------------------------ extensions

    def put_many(
        self, items: Sequence[tuple[str, Path | bytes]], message: str = "gpurelay batch"
    ) -> None:
        """Upload many files in one commit (``BatchStorage``)."""
        self._commit([self._add(remote, source) for remote, source in items], message)

    def delete_many(self, remotes: Iterable[str], message: str = "gpurelay delete") -> None:
        """Delete many keys in one commit; missing keys are ignored."""
        wanted = sorted(set(remotes))
        if not wanted:
            return
        info = self._call(
            lambda: self.api.get_paths_info(
                self.repo_id, wanted, repo_type=REPO_TYPE, token=self.token
            ),
            "paths info",
        )
        present = sorted({i.path for i in info})
        self._commit([self._delete(r) for r in present], message)

    def sha256(self, remote: str) -> str | None:
        """SHA-256 of the stored file: from LFS metadata when available, else downloaded."""
        validate_key(remote)
        info = self._call(
            lambda: self.api.get_paths_info(
                self.repo_id, [remote], repo_type=REPO_TYPE, token=self.token
            ),
            f"paths info {remote}",
        )
        if not info:
            return None
        lfs = getattr(info[0], "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha:
            return str(sha)
        with tempfile.TemporaryDirectory(prefix="gpurelay-sha-") as tmp:
            target = Path(tmp) / "data"
            try:
                self.get_file(remote, target)
            except StorageNotFoundError:
                return None
            return sha256_file(target)

    def used_bytes(self) -> int:
        """Total size of the files currently in the repo (history excluded)."""
        items = self._call(
            lambda: list(
                self.api.list_repo_tree(
                    self.repo_id, recursive=True, repo_type=REPO_TYPE, token=self.token
                )
            ),
            "list repo",
        )
        return sum(int(getattr(i, "size", 0) or 0) for i in items)

    def squash_history(self) -> None:
        """Drop git history so deleted files stop counting against storage (VERIFY)."""
        self._call(
            lambda: self.api.super_squash_history(
                self.repo_id, repo_type=REPO_TYPE, token=self.token
            ),
            "squash history",
        )


__all__ = ["REPO_TYPE", "HFHubStorage"]
