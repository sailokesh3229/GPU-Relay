"""Login for a dashboard served on the internet (``gpurelay serve``, plan W1).

Two roles: the owner (everything) and an optional viewer (read-only, e.g. a reviewer). The
passwords come from the environment, never the config file. A session is a signed cookie
(HMAC-SHA256, expiring), so the server keeps no session table; changing a password or the
session key logs everyone out. Failed logins are rate limited per client and in total.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from gpurelay.errors import ConfigError

ENV_OWNER_PASSWORD = "GPURELAY_OWNER_PASSWORD"
ENV_VIEWER_PASSWORD = "GPURELAY_VIEWER_PASSWORD"
ENV_SESSION_KEY = "GPURELAY_SESSION_KEY"
MIN_PASSWORD_LEN = 12
COOKIE_NAME = "gpurelay_session"
SESSION_S = 12 * 3600
OWNER = "owner"
VIEWER = "viewer"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass
class RateLimiter:
    """At most ``per_client`` failures per client and ``total`` overall in ``window_s``."""

    per_client: int = 5
    total: int = 50
    window_s: float = 900.0
    clock: Callable[[], float] = time.time
    _fails: dict[str, deque[float]] = field(default_factory=dict)
    _all: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _trim(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.window_s:
            q.popleft()

    def blocked(self, client: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._fails.get(client, deque())
            self._trim(q, now)
            self._trim(self._all, now)
            return len(q) >= self.per_client or len(self._all) >= self.total

    def failed(self, client: str) -> None:
        now = self.clock()
        with self._lock:
            self._fails.setdefault(client, deque()).append(now)
            self._all.append(now)
            if len(self._fails) > 10_000:  # forget the oldest clients, keep memory bounded
                for stale in list(self._fails)[:5_000]:
                    del self._fails[stale]


@dataclass
class Auth:
    owner_password: str
    viewer_password: str | None = None
    session_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    session_s: float = SESSION_S
    secure_cookie: bool = True
    clock: Callable[[], float] = time.time
    limiter: RateLimiter = field(default_factory=RateLimiter)

    def __post_init__(self) -> None:
        if len(self.owner_password) < MIN_PASSWORD_LEN:
            raise ConfigError(
                f"the owner password must have at least {MIN_PASSWORD_LEN} characters"
            )
        if self.viewer_password is not None:
            if len(self.viewer_password) < MIN_PASSWORD_LEN:
                raise ConfigError(
                    f"the viewer password must have at least {MIN_PASSWORD_LEN} characters"
                )
            if hmac.compare_digest(self.viewer_password, self.owner_password):
                raise ConfigError("the viewer password must differ from the owner password")
        # Bound to the passwords: changing one invalidates every session issued before.
        self._key = hmac.new(
            self.session_key,
            hashlib.sha256(
                f"{self.owner_password}\0{self.viewer_password or ''}".encode()
            ).digest(),
            hashlib.sha256,
        ).digest()

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Auth:
        env = dict(os.environ if env is None else env)
        owner = env.get(ENV_OWNER_PASSWORD)
        if not owner:
            raise ConfigError(
                f"set {ENV_OWNER_PASSWORD} (a secret): the dashboard is never served "
                "on the internet without a login"
            )
        key = env.get(ENV_SESSION_KEY)
        return cls(
            owner_password=owner,
            viewer_password=env.get(ENV_VIEWER_PASSWORD) or None,
            session_key=key.encode() if key else secrets.token_bytes(32),
        )

    def check_password(self, password: str) -> str | None:
        """Role for ``password``, or None. Both comparisons always run (constant time)."""
        given = password.encode()
        is_owner = hmac.compare_digest(given, self.owner_password.encode())
        is_viewer = self.viewer_password is not None and hmac.compare_digest(
            given, self.viewer_password.encode()
        )
        return OWNER if is_owner else VIEWER if is_viewer else None

    def _sign(self, payload: str) -> str:
        return _b64(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())

    def issue(self, role: str) -> str:
        body = _b64(json.dumps({"r": role, "exp": int(self.clock() + self.session_s)}).encode())
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str | None) -> str | None:
        """Role of a valid, unexpired session token, else None."""
        if not token or token.count(".") != 1:
            return None
        body, sig = token.split(".")
        if not hmac.compare_digest(sig, self._sign(body)):
            return None
        try:
            data = json.loads(_unb64(body))
            role, exp = data["r"], float(data["exp"])
        except (ValueError, KeyError, TypeError):
            return None
        if role not in (OWNER, VIEWER) or exp < self.clock():
            return None
        if role == VIEWER and self.viewer_password is None:
            return None  # viewer access was switched off
        return str(role)
