"""The website's account store on both databases it runs on: SQLite (tests, local runs) and
PostgreSQL (the live site). PostgreSQL runs when ``GPURELAY_TEST_PG_URL`` points at a
throwaway database (everything in it is deleted first); otherwise only SQLite runs."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpurelay.accounts.store import AccountError, UserStore

PW = "correct horse battery"
PG_URL = os.environ.get("GPURELAY_TEST_PG_URL", "")


def _wipe_pg(url: str) -> None:
    import psycopg

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")


@pytest.fixture(params=["sqlite", "postgres"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[UserStore]:
    if request.param == "sqlite":
        yield UserStore(f"sqlite:///{tmp_path / 'users.db'}")
        return
    if not PG_URL:
        pytest.skip("set GPURELAY_TEST_PG_URL to a throwaway PostgreSQL database")
    pytest.importorskip("psycopg")
    _wipe_pg(PG_URL)
    yield UserStore(PG_URL)


def test_accounts_sign_in_and_duplicates(store: UserStore) -> None:
    ada = store.create("Ada@Example.com", PW, username="Ada")
    assert store.count() == 1 and store.find("ada@example.com") == ada
    assert store.find_username("ada") == ada and store.get(ada.id) == ada
    assert store.authenticate("ADA@example.com", PW) == ada
    assert store.authenticate("ada", PW) == ada and store.authenticate("ada", "wrong") is None
    assert store.authenticate("nobody@example.com", PW) is None
    with pytest.raises(AccountError) as same_email:
        store.create("ada@example.com", PW, username="other")
    assert same_email.value.code == "exists"
    with pytest.raises(AccountError) as same_name:
        store.create("bob@example.com", PW, username="ADA")
    assert same_name.value.code == "taken"
    assert store.count() == 1  # nothing half created


def test_flags_usernames_and_passwords(store: UserStore) -> None:
    ada = store.create("ada@example.com", PW)
    bob = store.create("bob@example.com", PW, username="bob")
    store.set_email_verified(ada.id)
    store.set_notify(ada.id, False)
    store.set_active(ada.id, True)
    got = store.get(ada.id)
    assert got and got.email_verified and not got.notify and got.active
    assert [u.id for u in store.active_users()] == [ada.id]
    assert store.set_username(ada.id, "Ada_L") == "ada_l"
    with pytest.raises(AccountError):
        store.set_username(ada.id, "bob")
    epoch = store.get(ada.id).session_epoch  # type: ignore[union-attr]
    store.set_password(ada.id, "a different long password")
    after = store.get(ada.id)
    assert after and after.session_epoch == epoch + 1  # every session ends
    assert store.authenticate("ada_l", "a different long password") == after
    assert store.authenticate("ada_l", PW) is None and store.get(bob.id)


def test_secrets_settings_links_and_deleting_a_user(store: UserStore) -> None:
    ada = store.create("ada@example.com", PW)
    store.put_secret(ada.id, "hf_token", "sealed-1")
    store.put_secret(ada.id, "hf_token", "sealed-2")  # replaced, not duplicated
    store.put_secret(ada.id, "platform.runpod.api_key", "sealed-3")
    assert store.secret(ada.id, "hf_token") == "sealed-2"
    assert set(store.secret_names(ada.id)) == {"hf_token", "platform.runpod.api_key"}
    store.delete_secret(ada.id, "platform.runpod.api_key")
    assert set(store.secret_names(ada.id)) == {"hf_token"}
    store.put_settings(ada.id, '{"a": 1}')
    store.put_settings(ada.id, '{"a": 2}')
    assert store.settings(ada.id) == '{"a": 2}'
    now = store.clock()
    bob = store.create("bob@example.com", PW)
    store.put_link("hash-first", ada.id, "reset", now + 3600)
    store.put_link("hash-reset", ada.id, "reset", now + 3600)  # a new one replaces it
    store.put_link("hash-old", bob.id, "reset", now - 1)
    store.put_link("hash-helper", ada.id, "helper", now + 3600)
    assert store.take_link("hash-first", "reset") is None  # only the newest link works
    assert store.take_link("hash-old", "reset") is None  # expired
    assert store.take_link("hash-reset", "verify") is None  # wrong kind
    assert store.take_link("hash-reset", "reset") == ada.id
    assert store.take_link("hash-reset", "reset") is None  # once
    assert store.link_user("hash-helper", "helper") == ada.id  # kept: a helper key
    assert store.has_link(ada.id, "helper")
    store.delete_links(ada.id, "helper")
    assert not store.has_link(ada.id, "helper")
    store.put_meta("vault_check", "v1:abc")
    assert store.meta("vault_check") == "v1:abc"
    store.delete(ada.id)
    assert store.get(ada.id) is None and store.secret(ada.id, "hf_token") is None
    assert store.settings(ada.id) is None and store.count() == 1


def test_reopening_keeps_everything(store: UserStore, tmp_path: Path) -> None:
    ada = store.create("ada@example.com", PW, username="ada")
    url = PG_URL if store._sqlite is None else f"sqlite:///{tmp_path / 'users.db'}"
    again = UserStore(url)  # the schema and added columns are created only once
    assert again.find_username("ada") == ada and again.count() == 1
