"""Session setup survives transient storage errors (hardening campaign).

Found by test_faults: one failed read while a session started (bundle, checkpoint pointer)
ended the session as an infrastructure failure, and a few in a row failed the job.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from gpurelay.bundle.packager import upload_bundle
from gpurelay.errors import StorageError
from gpurelay.models import ExitCode
from gpurelay.runtime import entry
from gpurelay.storage import LocalFSStorage


class FirstCallFails:
    """Every storage operation fails the first time it is called."""

    def __init__(self, inner: LocalFSStorage) -> None:
        self._inner = inner
        self.failed: list[str] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            if name not in self.failed:
                self.failed.append(name)
                raise StorageError(f"injected fault in {name}")
            return attr(*args, **kwargs)

        return call


def test_setup_retries_transient_storage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalFSStorage(tmp_path / "store")
    script = tmp_path / "job.py"
    script.write_text("print('hello')\n")
    upload_bundle(store, "job-1", script, include_wheel=False)
    flaky = FirstCallFails(store)
    monkeypatch.setattr(entry, "open_storage", lambda uri: flaky)
    monkeypatch.setattr(
        entry, "SETUP_RETRY", dataclasses.replace(entry.SETUP_RETRY, sleep=lambda s: None)
    )

    code = entry.main(
        ["--job", "job-1", "--segment", "seg-1", "--storage", "unused",
         "--work-dir", str(tmp_path / "work"), "--runtime-dir", str(tmp_path / "runtime")]
    )  # fmt: skip

    assert code == int(ExitCode.COMPLETED)
    assert {"get_bytes", "put_bytes"} <= set(flaky.failed)
