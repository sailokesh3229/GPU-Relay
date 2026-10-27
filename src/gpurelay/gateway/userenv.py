"""A process environment that holds one user's tokens and nothing of the server's.

Every process that acts for a user (the setup checks in M2, the user's GPU Relay in M3)
starts from an allowlist of harmless variables, never from the server's own environment,
which holds the database URL, the vault key and the session key. Tokens go in environment
variables only (no token files): ``HF_TOKEN``, ``KAGGLE_API_TOKEN`` (or the older
``KAGGLE_USERNAME`` / ``KAGGLE_KEY``), ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``. Home and
config directories point inside the user's own folder, and the keyring is switched off, so
no library falls back to credentials of the server's account.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from gpurelay.config import Config, save_config
from gpurelay.constants import ENV_HOME

HF = "hf_token"
KAGGLE = "kaggle_token"
MODAL_ID = "modal_token_id"
MODAL_SECRET = "modal_token_secret"
SECRET_NAMES = (HF, KAGGLE, MODAL_ID, MODAL_SECRET)

SAFE_ENV = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
    "TMPDIR", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)  # fmt: skip


def kaggle_env(token: str) -> dict[str, str]:
    """A Kaggle token as environment variables: an API token, or a kaggle.json's contents."""
    token = token.strip()
    if token.startswith("{"):
        try:
            data = json.loads(token)
            return {"KAGGLE_USERNAME": str(data["username"]), "KAGGLE_KEY": str(data["key"])}
        except (ValueError, KeyError, TypeError):
            pass
    return {"KAGGLE_API_TOKEN": token}


def user_environment(
    home: Path, secrets: Mapping[str, str], base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Environment for a process acting for one user, rooted at ``home``."""
    base = os.environ if base is None else base
    env = {name: base[name] for name in SAFE_ENV if name in base}
    env.update({
        ENV_HOME: str(home),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "appdata"),
        "LOCALAPPDATA": str(home / "appdata"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "HF_HOME": str(home / ".cache" / "huggingface"),
        "KAGGLE_CONFIG_DIR": str(home / ".kaggle"),
        "MODAL_CONFIG_PATH": str(home / ".modal.toml"),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    })  # fmt: skip
    if secrets.get(HF):
        env["HF_TOKEN"] = secrets[HF]
    if secrets.get(KAGGLE):
        env.update(kaggle_env(secrets[KAGGLE]))
    if secrets.get(MODAL_ID) and secrets.get(MODAL_SECRET):
        env["MODAL_TOKEN_ID"] = secrets[MODAL_ID]
        env["MODAL_TOKEN_SECRET"] = secrets[MODAL_SECRET]
    return env


def server_rules(config: Config) -> Config:
    """What every user's GPU Relay on the server must follow: no jobs on the server's own CPU
    (submitted code never runs here), no desktop notifications, Lightning AI paused."""
    config = config.model_copy(deep=True)
    config.backends.local.enabled = False
    config.backends.lightning.enabled = False
    config.notify.desktop = False
    return config


def write_home(home: Path, config: Config) -> Path:
    """Create the user's folder with their config file (no secrets in it)."""
    home.mkdir(parents=True, exist_ok=True)
    for sub in ("appdata", ".config", ".cache", ".kaggle"):
        (home / sub).mkdir(exist_ok=True)
    return save_config(server_rules(config), home / "config.toml")
