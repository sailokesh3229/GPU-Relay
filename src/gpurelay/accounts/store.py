"""The user table, in Postgres on a server or SQLite for tests and local runs.

``DATABASE_URL`` picks the database: ``postgres://`` / ``postgresql://`` (psycopg, the
``server`` extra) or ``sqlite:///<path>``. The server's own disk is temporary, so a real
deployment uses Postgres. SQL here is the small subset both understand; placeholders are
written ``?`` and translated for psycopg.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurelay.accounts import passwords
from gpurelay.errors import GpuRelayError

ENV_DATABASE_URL = "DATABASE_URL"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_EMAIL = 254

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL,
        session_epoch INTEGER NOT NULL DEFAULT 0,
        email_verified INTEGER NOT NULL DEFAULT 0,
        active INTEGER NOT NULL DEFAULT 0,
        notify INTEGER NOT NULL DEFAULT 1
    )""",
    # Platform tokens, sealed by accounts/vault.py (never stored in plain text).
    """CREATE TABLE IF NOT EXISTS secrets (
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        sealed TEXT NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (user_id, name)
    )""",
    # The user's GPU Relay settings (config.toml as JSON; no secrets in it).
    """CREATE TABLE IF NOT EXISTS settings (
        user_id TEXT PRIMARY KEY,
        config TEXT NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    )""",
    # One-time links (email verification, password reset): only a hash of the token.
    """CREATE TABLE IF NOT EXISTS links (
        token_hash TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL
    )""",
)


ADDED_USER_COLUMNS = {
    "email_verified": "INTEGER NOT NULL DEFAULT 0",
    "active": "INTEGER NOT NULL DEFAULT 0",
    "notify": "INTEGER NOT NULL DEFAULT 1",
}
USER_COLUMNS = "id, email, created_at, session_epoch, email_verified, active, notify"


class AccountError(GpuRelayError):
    """A sign-up or account change was refused. ``code`` names the reason for the page."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class User:
    id: str
    email: str
    created_at: float
    session_epoch: int
    email_verified: bool = False
    active: bool = False  # has jobs that need its controller (started when the site starts)
    notify: bool = True  # email when a job needs the user, finishes or fails


def normalize_email(email: str) -> str:
    return email.strip().lower()


def email_problem(email: str) -> str | None:
    if len(email) > MAX_EMAIL or not EMAIL_RE.match(email):
        return "enter a valid email address"
    return None


class UserStore:
    def __init__(self, url: str, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        if url.startswith(("postgres://", "postgresql://")):
            import psycopg  # the "server" extra

            self._pg_url: str | None = url
            self._sqlite: sqlite3.Connection | None = None
            self._psycopg: Any = psycopg
        elif url.startswith("sqlite:///"):
            path = Path(url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pg_url = None
            self._sqlite = sqlite3.connect(str(path), check_same_thread=False)
        else:
            raise GpuRelayError(f"{ENV_DATABASE_URL} must start with postgresql:// or sqlite:///")
        for statement in SCHEMA:
            self._run(statement)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Columns added after a database was created (CREATE TABLE IF NOT EXISTS keeps the
        old table as it was)."""
        if self._sqlite is not None:
            have = {row[1] for row in self._all("PRAGMA table_info(users)")}
        else:
            have = {row[0] for row in self._all(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"
            )}  # fmt: skip
        for column, ddl in ADDED_USER_COLUMNS.items():
            if column not in have:
                self._run(f"ALTER TABLE users ADD COLUMN {column} {ddl}")

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        if self._sqlite is not None:
            with self._lock, self._sqlite:  # one writer; commit or roll back
                yield self._sqlite.cursor()
            return
        with (
            self._psycopg.connect(self._pg_url, connect_timeout=10) as conn,
            conn.cursor() as cur,
        ):
            yield cur

    def _sql(self, sql: str) -> str:
        return sql if self._sqlite is not None else sql.replace("?", "%s")

    def _run(self, sql: str, args: Sequence[Any] = ()) -> int:
        with self._cursor() as cur:
            cur.execute(self._sql(sql), tuple(args))
            return int(cur.rowcount or 0)

    def _one(self, sql: str, args: Sequence[Any] = ()) -> tuple[Any, ...] | None:
        with self._cursor() as cur:
            cur.execute(self._sql(sql), tuple(args))
            row = cur.fetchone()
            return tuple(row) if row is not None else None

    def _unique_violation(self, exc: Exception) -> bool:
        if isinstance(exc, sqlite3.IntegrityError):
            return True
        return self._sqlite is None and isinstance(exc, self._psycopg.errors.UniqueViolation)

    # ---------------------------------------------------------------- users

    def count(self) -> int:
        row = self._one("SELECT COUNT(*) FROM users")
        return int(row[0]) if row else 0

    def _user(self, row: tuple[Any, ...] | None) -> User | None:
        if row is None:
            return None
        return User(id=row[0], email=row[1], created_at=float(row[2]), session_epoch=int(row[3]),
                    email_verified=bool(row[4]), active=bool(row[5]),
                    notify=bool(row[6]))  # fmt: skip

    def get(self, user_id: str) -> User | None:
        return self._user(self._one(f"SELECT {USER_COLUMNS} FROM users WHERE id = ?", (user_id,)))

    def find(self, email: str) -> User | None:
        return self._user(
            self._one(
                f"SELECT {USER_COLUMNS} FROM users WHERE email = ?", (normalize_email(email),)
            )
        )

    def _all(self, sql: str, args: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._cursor() as cur:
            cur.execute(self._sql(sql), tuple(args))
            return [tuple(row) for row in cur.fetchall()]

    def active_users(self) -> list[User]:
        rows = self._all(f"SELECT {USER_COLUMNS} FROM users WHERE active = 1")
        return [u for u in map(self._user, rows) if u is not None]

    def set_active(self, user_id: str, active: bool) -> None:
        self._run("UPDATE users SET active = ? WHERE id = ?", (int(active), user_id))

    def set_notify(self, user_id: str, notify: bool) -> None:
        self._run("UPDATE users SET notify = ? WHERE id = ?", (int(notify), user_id))

    def set_email_verified(self, user_id: str) -> None:
        self._run("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))

    def delete(self, user_id: str) -> None:
        """The account, its tokens, settings and links (the user's own repos stay theirs)."""
        for table in ("secrets", "settings", "links"):
            self._run(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        self._run("DELETE FROM users WHERE id = ?", (user_id,))

    def create(self, email: str, password: str) -> User:
        email = normalize_email(email)
        if email_problem(email):
            raise AccountError(email_problem(email) or "", "email")
        if passwords.problem(password):
            raise AccountError(passwords.problem(password) or "", "password")
        user = User(id="u-" + secrets.token_hex(8), email=email, created_at=self.clock(),
                    session_epoch=0)  # fmt: skip
        try:
            self._run(
                "INSERT INTO users (id, email, password_hash, created_at, session_epoch) "
                "VALUES (?, ?, ?, ?, 0)",
                (user.id, email, passwords.hash_password(password), user.created_at),
            )
        except Exception as exc:
            if self._unique_violation(exc):
                raise AccountError("an account with this email already exists", "exists") from None
            raise
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        row = self._one(
            f"SELECT {USER_COLUMNS}, password_hash FROM users WHERE email = ?",
            (normalize_email(email),),
        )
        if row is None:
            passwords.dummy_check(password)
            return None
        if not passwords.verify_password(password, row[7]):
            return None
        return self._user(row[:7])

    def set_password(self, user_id: str, password: str) -> None:
        """New password; every existing session of the user ends."""
        if passwords.problem(password):
            raise AccountError(passwords.problem(password) or "", "password")
        self._run(
            "UPDATE users SET password_hash = ?, session_epoch = session_epoch + 1 WHERE id = ?",
            (passwords.hash_password(password), user_id),
        )

    def end_sessions(self, user_id: str) -> None:
        """Sign the user out everywhere."""
        self._run("UPDATE users SET session_epoch = session_epoch + 1 WHERE id = ?", (user_id,))

    # ---------------------------------------------------------------- tokens and settings

    def put_secret(self, user_id: str, name: str, sealed: str) -> None:
        self._run("DELETE FROM secrets WHERE user_id = ? AND name = ?", (user_id, name))
        self._run(
            "INSERT INTO secrets (user_id, name, sealed, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, name, sealed, self.clock()),
        )

    def secret(self, user_id: str, name: str) -> str | None:
        row = self._one(
            "SELECT sealed FROM secrets WHERE user_id = ? AND name = ?", (user_id, name)
        )
        return str(row[0]) if row else None

    def secret_names(self, user_id: str) -> dict[str, float]:
        """Name -> when it was saved (never the value)."""
        rows = self._all("SELECT name, updated_at FROM secrets WHERE user_id = ?", (user_id,))
        return {str(name): float(at) for name, at in rows}

    def delete_secret(self, user_id: str, name: str) -> None:
        self._run("DELETE FROM secrets WHERE user_id = ? AND name = ?", (user_id, name))

    def settings(self, user_id: str) -> str | None:
        row = self._one("SELECT config FROM settings WHERE user_id = ?", (user_id,))
        return str(row[0]) if row else None

    def put_settings(self, user_id: str, config_json: str) -> None:
        self._run("DELETE FROM settings WHERE user_id = ?", (user_id,))
        self._run(
            "INSERT INTO settings (user_id, config, updated_at) VALUES (?, ?, ?)",
            (user_id, config_json, self.clock()),
        )

    # ---------------------------------------------------------------- one-time links

    def put_link(self, token_hash: str, user_id: str, kind: str, expires_at: float) -> None:
        self._run("DELETE FROM links WHERE user_id = ? AND kind = ?", (user_id, kind))
        self._run(
            "INSERT INTO links (token_hash, user_id, kind, expires_at) VALUES (?, ?, ?, ?)",
            (token_hash, user_id, kind, expires_at),
        )

    def take_link(self, token_hash: str, kind: str) -> str | None:
        """The link's user id, once: the link is deleted; expired links count as missing."""
        row = self._one(
            "SELECT user_id, expires_at FROM links WHERE token_hash = ? AND kind = ?",
            (token_hash, kind),
        )
        if row is None:
            return None
        self._run("DELETE FROM links WHERE token_hash = ?", (token_hash,))
        return str(row[0]) if float(row[1]) >= self.clock() else None
