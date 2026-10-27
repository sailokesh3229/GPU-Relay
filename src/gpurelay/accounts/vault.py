"""Encryption of users' platform tokens at rest (plan M2).

AES-256-GCM with a key that exists only in the server's environment
(``GPURELAY_VAULT_KEY``: 32 random bytes, base64). Each ciphertext is bound to its user and
name (associated data), so a row copied to another user or another name does not decrypt.
Stored as ``v1:<base64 nonce + ciphertext>``; the version allows a later key rotation.
"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from collections.abc import Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gpurelay.errors import ConfigError, GpuRelayError

ENV_VAULT_KEY = "GPURELAY_VAULT_KEY"
PREFIX = "v1:"


class VaultError(GpuRelayError):
    """A stored token could not be decrypted (wrong key or tampered row)."""


def new_key() -> str:
    """A fresh key for ``GPURELAY_VAULT_KEY``."""
    return base64.b64encode(secrets.token_bytes(32)).decode("ascii")


class Vault:
    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ConfigError(f"{ENV_VAULT_KEY} must be 32 bytes, base64 encoded")
        self._aead = AESGCM(key)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Vault:
        env = os.environ if env is None else env
        raw = env.get(ENV_VAULT_KEY, "").strip()
        if not raw:
            raise ConfigError(
                f'set {ENV_VAULT_KEY} (create one with: python -c "from gpurelay.accounts.'
                'vault import new_key; print(new_key())")'
            )
        try:
            return cls(base64.b64decode(raw, validate=True))
        except binascii.Error:
            raise ConfigError(f"{ENV_VAULT_KEY} is not valid base64") from None

    @staticmethod
    def _aad(user_id: str, name: str) -> bytes:
        return f"gpurelay\0{user_id}\0{name}".encode()

    def seal(self, user_id: str, name: str, plaintext: str) -> str:
        nonce = secrets.token_bytes(12)
        sealed = self._aead.encrypt(nonce, plaintext.encode(), self._aad(user_id, name))
        return PREFIX + base64.b64encode(nonce + sealed).decode("ascii")

    def open(self, user_id: str, name: str, stored: str) -> str:
        if not stored.startswith(PREFIX):
            raise VaultError(f"unknown token format for {name}")
        try:
            raw = base64.b64decode(stored.removeprefix(PREFIX), validate=True)
            return self._aead.decrypt(raw[:12], raw[12:], self._aad(user_id, name)).decode()
        except (binascii.Error, InvalidTag, ValueError):
            raise VaultError(f"the stored {name} cannot be read (server key changed?)") from None
