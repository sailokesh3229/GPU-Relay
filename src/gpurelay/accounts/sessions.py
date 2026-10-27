"""Signed session cookies for user accounts (HMAC-SHA256, expiring, no session table).

A token carries the user id and the user's ``session_epoch``; the gateway compares the epoch
with the database, so a password change or "sign out everywhere" ends old sessions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

COOKIE_NAME = "gpurelay_user"
SESSION_S = 7 * 24 * 3600.0


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass
class SessionSigner:
    key: bytes
    session_s: float = SESSION_S
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if len(self.key) < 16:
            raise ValueError("the session key is too short")
        self._key = hmac.new(self.key, b"gpurelay user sessions", hashlib.sha256).digest()

    def _sign(self, body: str) -> str:
        return _b64(hmac.new(self._key, body.encode(), hashlib.sha256).digest())

    def issue(self, user_id: str, epoch: int) -> str:
        body = _b64(json.dumps(
            {"u": user_id, "e": epoch, "exp": int(self.clock() + self.session_s)}
        ).encode())  # fmt: skip
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str | None) -> tuple[str, int] | None:
        """(user id, epoch) of a valid, unexpired token, else None."""
        if not token or token.count(".") != 1 or len(token) > 1024:
            return None
        body, sig = token.split(".")
        if not hmac.compare_digest(sig, self._sign(body)):
            return None
        try:
            data = json.loads(_unb64(body))
            user_id, epoch, exp = str(data["u"]), int(data["e"]), float(data["exp"])
        except (ValueError, KeyError, TypeError):
            return None
        if exp < self.clock():
            return None
        return user_id, epoch
