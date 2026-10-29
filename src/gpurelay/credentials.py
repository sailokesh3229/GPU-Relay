"""Credentials: environment variables first, then the OS keyring (spec §13.1).

Secrets are never written to ``config.toml``. One credential set per service (hard rule 1).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

KEYRING_SERVICE = "gpurelay"
HF_TOKEN = "hf_token"
GDRIVE_CREDENTIALS = "gdrive_credentials"  # OAuth user credentials JSON


def _keyring() -> object | None:
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return None
    try:
        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
    except Exception:
        return None
    return keyring


def get_secret(name: str, env: str | None = None) -> str | None:
    """Value of secret ``name``: environment variable ``env`` if set, else the keyring."""
    if env and os.environ.get(env):
        return os.environ[env]
    kr = _keyring()
    if kr is None:
        return None
    try:
        value = kr.get_password(KEYRING_SERVICE, name)  # type: ignore[attr-defined]
    except Exception as exc:
        log.debug("keyring lookup failed: %s", exc)
        return None
    return str(value) if value else None


def set_secret(name: str, value: str) -> bool:
    """Store a secret in the OS keyring; False if no usable keyring exists."""
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.set_password(KEYRING_SERVICE, name, value)  # type: ignore[attr-defined]
    except Exception as exc:
        log.warning("could not store %s in the keyring: %s", name, exc)
        return False
    return True


def delete_secret(name: str) -> None:
    """Remove a secret from the OS keyring (nothing happens if it is not there)."""
    kr = _keyring()
    if kr is None:
        return
    try:
        kr.delete_password(KEYRING_SERVICE, name)  # type: ignore[attr-defined]
    except Exception as exc:  # PasswordDeleteError when absent
        log.debug("keyring delete of %s: %s", name, exc)


def hf_token() -> str | None:
    """Hugging Face token for the relay storage repo (``HF_TOKEN`` or keyring)."""
    return get_secret(HF_TOKEN, "HF_TOKEN")


def kaggle_credentials_present() -> bool:
    """Whether the official Kaggle client will find credentials (checked *without*
    importing it: ``import kaggle`` exits the process when there are none)."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    return any(
        (config_dir / name).is_file()
        for name in ("access_token", "kaggle.json", "credentials.json")
    )


def gdrive_credentials() -> dict[str, object] | None:
    """Google Drive OAuth credentials: ``$GPURELAY_GDRIVE_CREDENTIALS`` (JSON) or keyring."""
    import json

    raw = get_secret(GDRIVE_CREDENTIALS, env="GPURELAY_GDRIVE_CREDENTIALS")
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    return info if isinstance(info, dict) else None
