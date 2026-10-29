"""Secrets in the OS keyring (or environment variables), never in files GPU Relay writes."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from gpurelay import credentials


class FakeKeyring:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.fail = False

    def get_password(self, service: str, name: str) -> str | None:
        if self.fail:
            raise RuntimeError("locked")
        return self.store.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        if self.fail:
            raise RuntimeError("locked")
        self.store[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        if (service, name) not in self.store:
            raise KeyError(name)  # keyring raises PasswordDeleteError
        del self.store[(service, name)]


@pytest.fixture
def ring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "_keyring", lambda: fake)
    return fake


def test_keyring_round_trip_and_environment_first(
    ring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert credentials.hf_token() is None
    assert credentials.set_secret(credentials.HF_TOKEN, "hf_from_keyring")
    assert credentials.hf_token() == "hf_from_keyring"
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert credentials.hf_token() == "hf_from_env"  # the environment wins
    credentials.delete_secret(credentials.HF_TOKEN)
    credentials.delete_secret(credentials.HF_TOKEN)  # absent: nothing happens
    monkeypatch.delenv("HF_TOKEN")
    assert credentials.hf_token() is None


def test_a_broken_or_missing_keyring_is_not_an_error(
    ring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    ring.fail = True
    assert credentials.get_secret("x") is None and credentials.set_secret("x", "v") is False
    monkeypatch.setattr(credentials, "_keyring", lambda: None)
    assert credentials.get_secret("x") is None and credentials.set_secret("x", "v") is False
    credentials.delete_secret("x")


def test_no_usable_keyring_backend_means_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fail_mod = types.ModuleType("keyring.backends.fail")

    class Keyring:
        pass

    fail_mod.Keyring = Keyring  # type: ignore[attr-defined]
    keyring_mod: Any = types.ModuleType("keyring")
    keyring_mod.get_keyring = lambda: Keyring()
    backends = types.ModuleType("keyring.backends")
    backends.fail = fail_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring_mod)
    monkeypatch.setitem(sys.modules, "keyring.backends", backends)
    monkeypatch.setitem(sys.modules, "keyring.backends.fail", fail_mod)
    assert credentials._keyring() is None


def test_kaggle_credentials_are_found_without_importing_kaggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert not credentials.kaggle_credentials_present()
    (tmp_path / "access_token").write_text("KGAT_x")
    assert credentials.kaggle_credentials_present()
    (tmp_path / "access_token").unlink()
    monkeypatch.setenv("KAGGLE_USERNAME", "ada")
    assert not credentials.kaggle_credentials_present()  # needs the key too
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert credentials.kaggle_credentials_present()


def test_drive_credentials_must_be_a_json_object(
    ring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GPURELAY_GDRIVE_CREDENTIALS", raising=False)
    assert credentials.gdrive_credentials() is None
    monkeypatch.setenv("GPURELAY_GDRIVE_CREDENTIALS", "not json")
    assert credentials.gdrive_credentials() is None
    monkeypatch.setenv("GPURELAY_GDRIVE_CREDENTIALS", "[1, 2]")
    assert credentials.gdrive_credentials() is None
    monkeypatch.setenv("GPURELAY_GDRIVE_CREDENTIALS", '{"refresh_token": "r"}')
    assert credentials.gdrive_credentials() == {"refresh_token": "r"}
