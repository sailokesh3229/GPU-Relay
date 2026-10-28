"""Each user's tokens and settings, and the setup checks that change them (plan M2).

Tokens are sealed by the vault before they reach the database and opened only to build the
environment of a process that acts for their owner. A setup check runs in its own process
(:mod:`gpurelay.gateway.setup_worker`) in a throwaway folder; on success the gateway keeps
the updated settings and the new tokens. A new token is stored only after the platform
accepted it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurelay.accounts.store import UserStore
from gpurelay.accounts.vault import Vault, VaultError
from gpurelay.config import Config, load_config
from gpurelay.errors import GpuRelayError
from gpurelay.gateway.userenv import (
    HF,
    KAGGLE,
    MODAL_SECRET,
    SECRET_NAMES,
    server_rules,
    user_environment,
    write_home,
)

SETUP_TIMEOUT_S = 180.0
VAULT_CHECK = "vault_check"
WORKER = [sys.executable, "-m", "gpurelay.gateway.setup_worker"]

# (argv, env, stdin, timeout) -> stdout
Runner = Callable[[list[str], dict[str, str], str, float], str]


def run_process(argv: list[str], env: dict[str, str], stdin: str, timeout: float) -> str:
    try:
        done = subprocess.run(argv, input=stdin, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)  # fmt: skip
    except subprocess.TimeoutExpired:
        raise GpuRelayError("the platform did not answer in time; try again") from None
    return done.stdout


def scrub(text: str, secrets: Mapping[str, str]) -> str:
    """``text`` with every token value replaced (error messages must never quote one)."""
    for value in secrets.values():
        if value and len(value) >= 6:
            text = text.replace(value, "***")
    return text


@dataclass
class UserSpaces:
    store: UserStore
    vault: Vault
    root: Path  # the users' folders on this server
    runner: Runner = run_process

    # ---------------------------------------------------------------- tokens and settings

    def secrets(self, user_id: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for name in SECRET_NAMES:
            sealed = self.store.secret(user_id, name)
            if sealed is None:
                continue
            try:
                found[name] = self.vault.open(user_id, name, sealed)
            except VaultError:
                continue  # unreadable (key changed): treated as missing, the user adds it again
        return found

    def saved(self, user_id: str) -> dict[str, float]:
        return self.store.secret_names(user_id)

    def unreadable(self, user_id: str) -> list[str]:
        """Tokens that are stored but cannot be opened (the vault key changed)."""
        names = []
        for name in SECRET_NAMES:
            sealed = self.store.secret(user_id, name)
            if sealed is None:
                continue
            try:
                self.vault.open(user_id, name, sealed)
            except VaultError:
                names.append(name)
        return names

    def check_vault(self) -> bool:
        """True if this server's vault key is the one the stored tokens were sealed with.

        The first start stores a small value sealed with the key; every later start opens
        it. False means ``GPURELAY_VAULT_KEY`` changed: every stored token is unreadable,
        and users would have to add them again. Restoring the old key brings them back."""
        sealed = self.store.meta(VAULT_CHECK)
        if sealed is None:
            self.store.put_meta(VAULT_CHECK, self.vault.seal("site", VAULT_CHECK, "ok"))
            return True
        try:
            return self.vault.open("site", VAULT_CHECK, sealed) == "ok"
        except VaultError:
            return False

    def config(self, user_id: str) -> Config:
        raw = self.store.settings(user_id)
        return server_rules(Config.model_validate_json(raw) if raw else Config())

    def save_config(self, user_id: str, config: Config) -> None:
        self.store.put_settings(user_id, server_rules(config).model_dump_json(exclude_none=True))

    def ready(self, user_id: str) -> bool:
        """Storage connected and at least one way to get GPUs (Kaggle, Modal or Colab)."""
        config, unreadable = self.config(user_id), set(self.unreadable(user_id))
        saved = {k: v for k, v in self.saved(user_id).items() if k not in unreadable}
        backends = config.backends
        storage = (
            HF in saved and config.storage.kind == "hf_hub" and bool(config.storage.hf_repo_id)
        )
        gpus = (
            (KAGGLE in saved and backends.kaggle.enabled)
            or (MODAL_SECRET in saved and backends.modal.enabled)
            or backends.colab_manual.enabled
        )
        return storage and gpus

    def put_secrets(self, user_id: str, values: Mapping[str, str]) -> None:
        for name, value in values.items():
            self.store.put_secret(user_id, name, self.vault.seal(user_id, name, value))

    def remove_secrets(self, user_id: str, names: tuple[str, ...]) -> None:
        for name in names:
            self.store.delete_secret(user_id, name)

    # ---------------------------------------------------------------- checks

    def run_step(
        self, user_id: str, step: str, request: dict[str, Any], new: Mapping[str, str]
    ) -> dict[str, Any]:
        """Run one setup check with the user's tokens plus ``new``; keep both on success."""
        secrets = {**self.secrets(user_id), **new}
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="setup-", dir=self.root) as tmp:
            home = Path(tmp)
            write_home(home, self.config(user_id))
            out = self.runner(WORKER, user_environment(home, secrets),
                              json.dumps({**request, "step": step}), SETUP_TIMEOUT_S)  # fmt: skip
            result = _last_json(out)
            if "error" in result or not result.get("ok"):
                raise GpuRelayError(scrub(str(result.get("error", "setup failed")), secrets))
            config = load_config(home / "config.toml")
        self.save_config(user_id, config)
        self.put_secrets(user_id, new)
        return {k: v for k, v in result.items() if k != "ok"}


def _last_json(out: str) -> dict[str, Any]:
    for line in reversed(out.strip().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {"error": "the setup check did not finish"}
