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
from gpurelay.gateway.userenv import SECRET_NAMES, server_rules, user_environment, write_home

SETUP_TIMEOUT_S = 180.0
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

    def config(self, user_id: str) -> Config:
        raw = self.store.settings(user_id)
        return server_rules(Config.model_validate_json(raw) if raw else Config())

    def save_config(self, user_id: str, config: Config) -> None:
        self.store.put_settings(user_id, server_rules(config).model_dump_json(exclude_none=True))

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
