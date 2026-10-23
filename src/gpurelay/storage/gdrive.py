"""Google Drive storage (Phase 9 item 5): one app-created folder holds the whole store.

Keys map to folders and files under the store folder (``jobs/<job>/status.json`` ->
``jobs`` / ``<job>`` / ``status.json``). Least privilege: the OAuth scope is ``drive.file``,
so GPU Relay sees only files it created.

* **Atomic small writes:** writing an existing key replaces the file's *content* (a new
  revision of the same file), so readers see the old or the new bytes, never a mix.
* Drive allows several files with one name in a folder; if a race creates duplicates, the
  newest wins and the others are deleted on the next write.
* Errors: 404 -> :class:`StorageNotFoundError`; rate limits (403/429) and 5xx are retried
  with backoff.

Drive is reachable from the controller and from local sessions; Kaggle, Colab, and Lightning
sessions still need Hugging Face storage (their bootstraps fetch the runtime from the Hub).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

from gpurelay.errors import StorageError, StorageNotFoundError
from gpurelay.retry import RetryPolicy
from gpurelay.storage.base import validate_key, validate_prefix

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_FOLDER = "gpurelay-store"
_RETRY = RetryPolicy(attempts=6, base_delay_s=2.0, max_delay_s=60.0)
_T = TypeVar("_T")


class DriveApi(Protocol):
    """The Drive calls GPU Relay needs (``OfficialDriveApi`` wraps the official client).

    Entries are dicts with ``id``, ``name``, ``mimeType``, ``modifiedTime`` and, for files,
    optionally ``sha256Checksum`` and ``size``.
    """

    def children(self, parent_id: str, name: str | None = None) -> list[dict[str, Any]]: ...

    def create_folder(self, parent_id: str, name: str) -> str: ...

    def upload(self, parent_id: str, name: str, data: bytes, file_id: str | None) -> str: ...

    def download(self, file_id: str) -> bytes: ...

    def delete(self, file_id: str) -> None: ...


def _status(exc: BaseException) -> int | None:
    resp = getattr(exc, "resp", None)
    status = getattr(resp, "status", None) if resp is not None else None
    if status is None:
        status = getattr(exc, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


class OfficialDriveApi:
    """Drive API v3 through ``google-api-python-client`` (optional extra ``gdrive``)."""

    def __init__(self, credentials: Any) -> None:
        from googleapiclient.discovery import build

        self._files = build("drive", "v3", credentials=credentials, cache_discovery=False).files()

    @staticmethod
    def _quote(name: str) -> str:
        return name.replace("\\", "\\\\").replace("'", "\\'")

    def children(self, parent_id: str, name: str | None = None) -> list[dict[str, Any]]:
        query = f"'{parent_id}' in parents and trashed = false"
        if name is not None:
            query += f" and name = '{self._quote(name)}'"
        entries: list[dict[str, Any]] = []
        token = None
        while True:
            page = self._files.list(
                q=query,
                spaces="drive",
                pageSize=1000,
                pageToken=token,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, "
                "sha256Checksum)",
            ).execute()
            entries.extend(page.get("files", []))
            token = page.get("nextPageToken")
            if not token:
                return entries

    def create_folder(self, parent_id: str, name: str) -> str:
        body = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
        return str(self._files.create(body=body, fields="id").execute()["id"])

    def upload(self, parent_id: str, name: str, data: bytes, file_id: str | None) -> str:
        from googleapiclient.http import MediaIoBaseUpload

        media = MediaIoBaseUpload(
            io.BytesIO(data), mimetype="application/octet-stream", resumable=len(data) > 5 << 20
        )
        if file_id is not None:
            return str(self._files.update(fileId=file_id, media_body=media, fields="id")
                       .execute()["id"])  # fmt: skip
        body = {"name": name, "parents": [parent_id]}
        return str(self._files.create(body=body, media_body=media, fields="id").execute()["id"])

    def download(self, file_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, self._files.get_media(fileId=file_id))
        done = False
        while not done:
            _progress, done = downloader.next_chunk()
        return buffer.getvalue()

    def delete(self, file_id: str) -> None:
        self._files.delete(fileId=file_id).execute()


class GoogleDriveStorage:
    """:class:`~gpurelay.storage.base.Storage` in a Google Drive folder."""

    def __init__(self, folder_id: str, api: DriveApi, *, retry: RetryPolicy = _RETRY) -> None:
        self.folder_id = folder_id
        self.api = api
        self.retry = retry
        self._folders: dict[str, str] = {"": folder_id}  # folder path -> id

    def __repr__(self) -> str:
        return f"GoogleDriveStorage({self.folder_id!r})"

    # ------------------------------------------------------------------ plumbing

    def _call(self, fn: Callable[[], _T], what: str) -> _T:
        def attempt() -> _T:
            try:
                return fn()
            except (StorageError, FileNotFoundError):
                raise
            except Exception as exc:
                if _status(exc) == 404:
                    raise StorageNotFoundError(f"{what}: not found") from exc
                raise StorageError(f"{what}: {type(exc).__name__}: {exc}") from exc

        return self.retry.call(attempt, what=f"gdrive {what}")

    def _children(self, folder_id: str, path: str, name: str | None = None) -> list[dict[str, Any]]:
        """Entries of a folder; a folder that no longer exists is empty (cache dropped)."""
        try:
            return self._call(partial(self.api.children, folder_id, name), f"list {path}/")
        except StorageNotFoundError:
            for cached in [
                p for p in self._folders if p and (p == path or p.startswith(path + "/"))
            ]:
                del self._folders[cached]
            return []

    def _folder(self, path: str, *, create: bool) -> str | None:
        """Id of folder ``path`` ("a/b"), created when ``create``; None if missing."""
        if path in self._folders:
            return self._folders[path]
        parent_path, _, name = path.rpartition("/")
        parent = self._folder(parent_path, create=create)
        if parent is None:
            return None
        found = [
            e for e in self._children(parent, parent_path, name) if e.get("mimeType") == FOLDER_MIME
        ]
        if found:
            folder_id = str(sorted(found, key=lambda e: e.get("modifiedTime", ""))[0]["id"])
        elif create:
            folder_id = self._call(lambda: self.api.create_folder(parent, name), f"mkdir {path}")
        else:
            return None
        self._folders[path] = folder_id
        return folder_id

    def _files(self, remote: str) -> tuple[str | None, list[dict[str, Any]]]:
        """(parent folder id, files named like ``remote``, newest first)."""
        folder_path, _, name = validate_key(remote).rpartition("/")
        parent = self._folder(folder_path, create=False)
        if parent is None:
            return None, []
        found = [
            e for e in self._children(parent, folder_path, name) if e.get("mimeType") != FOLDER_MIME
        ]
        return parent, sorted(found, key=lambda e: e.get("modifiedTime", ""), reverse=True)

    # ------------------------------------------------------------------ Storage API

    def put_bytes(self, data: bytes, remote: str) -> None:
        folder_path, _, name = validate_key(remote).rpartition("/")
        parent = self._folder(folder_path, create=True)
        assert parent is not None
        _parent, existing = self._files(remote)
        target = existing[0]["id"] if existing else None
        self._call(lambda: self.api.upload(parent, name, data, target), f"put {remote}")
        for extra in existing[1:]:  # duplicates from a concurrent create: newest wins
            self._call(partial(self.api.delete, extra["id"]), f"dedupe {remote}")

    def put_file(self, local: Path, remote: str) -> None:
        self.put_bytes(Path(local).read_bytes(), remote)

    def get_bytes(self, remote: str) -> bytes | None:
        _parent, found = self._files(remote)
        if not found:
            return None
        try:
            return self._call(lambda: self.api.download(found[0]["id"]), f"get {remote}")
        except StorageNotFoundError:
            return None

    def get_file(self, remote: str, local: Path) -> None:
        data = self.get_bytes(remote)
        if data is None:
            raise StorageNotFoundError(f"{remote}: not found")
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=local.parent, delete=False) as fh:
            fh.write(data)
        os.replace(fh.name, local)

    def list(self, prefix: str) -> list[str]:
        validate_prefix(prefix)
        base = prefix.rpartition("/")[0]
        folder = self._folder(base, create=False) if base else self.folder_id
        if folder is None:
            return []
        keys: list[str] = []
        todo = [(base, folder)]
        while todo:
            path, folder_id = todo.pop()
            for entry in self._children(folder_id, path):
                key = f"{path}/{entry['name']}" if path else str(entry["name"])
                if entry.get("mimeType") == FOLDER_MIME:
                    self._folders.setdefault(key, str(entry["id"]))
                    todo.append((key, str(entry["id"])))
                elif key.startswith(prefix):
                    keys.append(key)
        return sorted(set(keys))

    def delete(self, remote: str) -> None:
        _parent, found = self._files(remote)
        for entry in found:
            with contextlib.suppress(StorageNotFoundError):
                self._call(partial(self.api.delete, entry["id"]), f"delete {remote}")

    def exists(self, remote: str) -> bool:
        return bool(self._files(remote)[1])

    def sha256(self, remote: str) -> str | None:
        """Drive's own SHA-256 of the stored content (VERIFY: present for binary files)."""
        _parent, found = self._files(remote)
        if not found:
            return None
        digest = found[0].get("sha256Checksum")
        return str(digest) if digest else None


# ====================================================================== auth


def credentials_from_info(info: dict[str, Any]) -> Any:
    """OAuth user credentials (with refresh token) from their JSON form."""
    from google.oauth2.credentials import Credentials

    return Credentials.from_authorized_user_info(info, SCOPES)


def authorize(client_secret_file: Path) -> dict[str, Any]:
    """One-time browser sign-in (OAuth "Desktop app" client). Returns the credentials JSON."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), SCOPES)
    creds = flow.run_local_server(port=0)
    info: dict[str, Any] = json.loads(creds.to_json())
    return info


def open_gdrive(folder_id: str, info: dict[str, Any]) -> GoogleDriveStorage:
    return GoogleDriveStorage(folder_id, OfficialDriveApi(credentials_from_info(info)))
