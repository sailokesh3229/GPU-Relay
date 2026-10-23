"""``gpurelay init``: storage repo, credentials, and the Kaggle secrets dataset (spec §15).

Hard rules: one account per service; the storage repo must be private; the token used by
Kaggle sessions lives only in the user's *private* Kaggle dataset (never in kernel source).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurelay.backends.kaggle import SECRETS_FILE, KaggleClient
from gpurelay.config import Config
from gpurelay.errors import ConfigError, HardRuleViolation
from gpurelay.storage.hf_hub import REPO_TYPE

DEFAULT_REPO = "gpurelay-store"
SECRETS_SLUG = "gpurelay-secrets"


@dataclass
class StorageSetup:
    repo_id: str
    hf_user: str
    created: bool


def setup_storage(config: Config, api: Any, *, repo: str | None = None) -> StorageSetup:
    """Create (or verify) the private dataset repo and point the config at it."""
    user = api.whoami()["name"]
    repo_id = repo if repo and "/" in repo else f"{user}/{repo or DEFAULT_REPO}"
    try:
        api.repo_info(repo_id, repo_type=REPO_TYPE)
        created = False
    except Exception:
        api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
        created = True
    info = api.repo_info(repo_id, repo_type=REPO_TYPE)
    if not getattr(info, "private", False):
        raise HardRuleViolation(
            f"{repo_id} is public: GPU Relay stores your code, data, and checkpoints there and "
            "requires a private repo"
        )
    config.storage.kind = "hf_hub"
    config.storage.hf_repo_id = repo_id
    return StorageSetup(repo_id=repo_id, hf_user=str(user), created=created)


@dataclass
class KaggleSetup:
    username: str
    secrets_dataset: str
    created: bool


def validate_session_token(token: str, repo_id: str, api: Any) -> str:
    """The token Kaggle sessions use must be a Hugging Face token that can see ``repo_id``.

    Returns the token owner's name. Raises ConfigError with a clear message otherwise.
    """
    if not token.startswith("hf_"):
        hint = " (that looks like a Kaggle token)" if token.startswith("KGAT") else ""
        raise ConfigError(f"the session token must be a Hugging Face token (hf_...){hint}")
    try:
        owner = api.whoami(token=token)["name"]
        api.repo_info(repo_id, repo_type=REPO_TYPE, token=token)
    except Exception as exc:
        raise ConfigError(f"the Hugging Face token cannot access {repo_id}: {exc}") from exc
    return str(owner)


def setup_kaggle(
    config: Config,
    client: KaggleClient,
    *,
    session_token: str,
    repo_id: str,
    rotate: bool = False,
) -> KaggleSetup:
    """Enable the Kaggle backend and create the private secrets dataset if missing.

    ``session_token`` should be a fine-grained Hugging Face token with write access to
    ``repo_id`` only: it is readable by every Kaggle session of this account.
    """
    if not session_token:
        raise ConfigError("a Hugging Face token for Kaggle sessions is required")
    username = client.username
    existing = config.backends.kaggle.username
    if existing and existing != username:
        raise HardRuleViolation(
            f"Kaggle is configured for {existing!r}; GPU Relay uses one account per service "
            f"(found credentials for {username!r})"
        )
    ref = f"{username}/{SECRETS_SLUG}"
    created = False
    exists = client.dataset_exists(ref)
    if not exists or rotate:
        with tempfile.TemporaryDirectory(prefix="gpurelay-secrets-") as tmp:
            folder = Path(tmp)
            (folder / SECRETS_FILE).write_text(
                json.dumps({"hf_token": session_token, "repo_id": repo_id}), encoding="utf-8"
            )
            (folder / "dataset-metadata.json").write_text(
                json.dumps(
                    {"title": SECRETS_SLUG, "id": ref, "licenses": [{"name": "other"}]}, indent=2
                ),
                encoding="utf-8",
            )
            if exists:
                client.replace_dataset(folder)  # new version, old versions deleted
            else:
                client.create_dataset(folder)  # private (public=False)
        created = True
    kaggle = config.backends.kaggle
    kaggle.enabled = True
    kaggle.username = username
    kaggle.secrets_dataset = ref
    return KaggleSetup(username=username, secrets_dataset=ref, created=created)


def setup_gdrive(config: Config, api: Any, *, folder: str = "gpurelay-store") -> str:
    """Create (or reuse) the store folder in the user's Drive and point the config at it.

    ``api`` is a :class:`~gpurelay.storage.gdrive.DriveApi`; with the ``drive.file`` scope it
    sees only folders GPU Relay created, so an existing one is ours.
    """
    from gpurelay.storage.gdrive import FOLDER_MIME

    found = [e for e in api.children("root", folder) if e.get("mimeType") == FOLDER_MIME]
    folder_id = str(found[0]["id"]) if found else api.create_folder("root", folder)
    config.storage.kind = "gdrive"
    config.storage.gdrive_folder_id = folder_id
    return folder_id
