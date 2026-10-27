"""Password hashing with scrypt (standard library only).

Stored as ``scrypt$<n>$<r>$<p>$<salt>$<hash>`` (base64), so the cost can be raised later
without breaking older hashes. Passwords are length-limited: scrypt's work grows with the
input, and a megabyte "password" must not tie up the server.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

MIN_LEN = 12
MAX_LEN = 256
N, R, P = 2**14, 8, 1  # ~16 MB and ~50 ms per hash
_DUMMY = None  # a hash to check against when the email is unknown (same timing)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32,
                          maxmem=64 * 1024 * 1024)  # fmt: skip


def problem(password: str) -> str | None:
    """Why ``password`` is not acceptable for a new account, or None."""
    if len(password) < MIN_LEN:
        return f"use at least {MIN_LEN} characters"
    if len(password) > MAX_LEN:
        return f"use at most {MAX_LEN} characters"
    return None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = _derive(password, salt, N, R, P)
    return f"scrypt${N}${R}${P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """True if ``password`` matches ``stored``; False for any malformed hash."""
    if len(password) > MAX_LEN:
        return False
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = _derive(password, base64.b64decode(salt), int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def dummy_check(password: str) -> None:
    """Spend the same time as a real check (unknown email: no timing difference)."""
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_password(secrets.token_urlsafe(16))
    verify_password(password[:MAX_LEN], _DUMMY)
