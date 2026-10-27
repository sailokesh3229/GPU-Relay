from __future__ import annotations

from pathlib import Path

import pytest

from gpurelay.accounts import passwords
from gpurelay.accounts.sessions import SessionSigner
from gpurelay.accounts.store import AccountError, UserStore
from gpurelay.errors import GpuRelayError

GOOD = "correct horse battery"


def _store(tmp_path: Path) -> UserStore:
    return UserStore(f"sqlite:///{tmp_path / 'users.db'}")


def test_password_hashes_are_salted_and_verify() -> None:
    a, b = passwords.hash_password(GOOD), passwords.hash_password(GOOD)
    assert a != b and a.startswith("scrypt$") and GOOD not in a
    assert passwords.verify_password(GOOD, a) and passwords.verify_password(GOOD, b)
    assert not passwords.verify_password(GOOD + "!", a)
    assert not passwords.verify_password(GOOD, "not a hash")
    assert not passwords.verify_password("x" * 10_000, a)  # too long: refused before scrypt
    assert passwords.problem("short") and passwords.problem("x" * 300)
    assert passwords.problem(GOOD) is None


def test_create_and_authenticate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    user = store.create("  Ada@Example.COM ", GOOD)
    assert user.email == "ada@example.com" and store.count() == 1
    assert store.authenticate("ADA@example.com", GOOD) == user
    assert store.authenticate("ada@example.com", "wrong password!") is None
    assert store.authenticate("nobody@example.com", GOOD) is None
    assert store.get(user.id) == user and store.get("u-missing") is None


@pytest.mark.parametrize(
    ("email", "password", "code"),
    [("not-an-email", GOOD, "email"), ("a@b.co", "short", "password")],
)
def test_bad_sign_ups_are_refused(tmp_path: Path, email: str, password: str, code: str) -> None:
    with pytest.raises(AccountError) as err:
        _store(tmp_path).create(email, password)
    assert err.value.code == code and _store(tmp_path).count() == 0


def test_one_account_per_email(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.create("ada@example.com", GOOD)
    with pytest.raises(AccountError) as err:
        store.create("ADA@example.com", "another long password")
    assert err.value.code == "exists" and store.count() == 1


def test_password_change_and_sign_out_everywhere_bump_the_epoch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    user = store.create("ada@example.com", GOOD)
    store.set_password(user.id, "a brand new passphrase")
    assert store.authenticate("ada@example.com", GOOD) is None
    changed = store.authenticate("ada@example.com", "a brand new passphrase")
    assert changed is not None and changed.session_epoch == 1
    store.end_sessions(user.id)
    assert (store.get(user.id) or changed).session_epoch == 2


def test_accounts_survive_reopening_the_database(tmp_path: Path) -> None:
    user = _store(tmp_path).create("ada@example.com", GOOD)
    assert _store(tmp_path).authenticate("ada@example.com", GOOD) == user


def test_only_postgres_or_sqlite_urls() -> None:
    with pytest.raises(GpuRelayError):
        UserStore("mysql://somewhere/db")


def test_session_tokens_are_signed_and_expire() -> None:
    now = [1000.0]
    signer = SessionSigner(b"k" * 32, session_s=60, clock=lambda: now[0])
    token = signer.issue("u-1", 3)
    assert signer.verify(token) == ("u-1", 3)
    body, sig = token.split(".")
    assert signer.verify(body + "." + sig[:-2] + "AA") is None
    assert SessionSigner(b"x" * 32, clock=lambda: now[0]).verify(token) is None  # other key
    assert signer.verify(None) is None and signer.verify("junk") is None
    now[0] += 61
    assert signer.verify(token) is None
    with pytest.raises(ValueError):
        SessionSigner(b"short")


def test_an_older_database_gets_the_new_columns(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # the M1 table, before verification and notices
        conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, "
                     "password_hash TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL, "
                     "session_epoch INTEGER NOT NULL DEFAULT 0)")  # fmt: skip
    store = UserStore(f"sqlite:///{path}")
    user = store.create("ada@example.com", GOOD)
    assert store.authenticate("ada@example.com", GOOD) == user and user.notify
