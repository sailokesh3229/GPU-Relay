"""Build a :class:`Storage` from a URI passed between controller and runtime.

* ``localfs:<path>`` — local directory (tests, local-only use);
* ``hf:<owner>/<repo>`` — private Hugging Face dataset repo; the token comes from
  ``HF_TOKEN`` or the OS keyring (never from the URI);
* ``gdrive:<folder id>`` — Google Drive folder; OAuth credentials come from
  ``GPURELAY_GDRIVE_CREDENTIALS`` or the OS keyring.
"""

from __future__ import annotations

from pathlib import Path

from gpurelay.storage.base import Storage
from gpurelay.storage.localfs import LocalFSStorage


def storage_uri(storage: Storage) -> str:
    """URI that reopens ``storage`` in another process."""
    if isinstance(storage, LocalFSStorage):
        return f"localfs:{storage.root}"
    from gpurelay.storage.hf_hub import HFHubStorage

    if isinstance(storage, HFHubStorage):
        return f"hf:{storage.repo_id}"
    from gpurelay.storage.gdrive import GoogleDriveStorage

    if isinstance(storage, GoogleDriveStorage):
        return f"gdrive:{storage.folder_id}"
    raise ValueError(f"no URI form for {type(storage).__name__}")


def open_storage(uri: str) -> Storage:
    """Open the storage named by ``uri``."""
    kind, sep, rest = uri.partition(":")
    if not sep or not rest:
        raise ValueError(f"invalid storage URI {uri!r}")
    if kind == "localfs":
        return LocalFSStorage(Path(rest))
    if kind == "hf":
        from gpurelay.credentials import hf_token
        from gpurelay.storage.hf_hub import HFHubStorage

        return HFHubStorage(rest, token=hf_token())
    if kind == "gdrive":
        from gpurelay.credentials import gdrive_credentials
        from gpurelay.storage.gdrive import open_gdrive

        info = gdrive_credentials()
        if info is None:
            raise ValueError("no Google Drive credentials: run `gpurelay init --gdrive`")
        return open_gdrive(rest, info)
    raise ValueError(f"unsupported storage kind {kind!r} in {uri!r}")
