from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from gpurelay.accounts.sessions import SessionSigner
from gpurelay.accounts.store import UserStore
from gpurelay.accounts.vault import Vault, new_key
from gpurelay.errors import ConfigError
from gpurelay.gateway.account_routes import token_hash
from gpurelay.gateway.app import GatewaySettings, create_gateway
from gpurelay.gateway.mail import Mailer
from gpurelay.gateway.notices import Notices
from gpurelay.gateway.space import UserSpaces

PW = "correct horse battery"
NEW = "a brand new passphrase"


class FakeMailer(Mailer):
    def __init__(self) -> None:
        super().__init__("smtp.example", 587, None, None, "relay@example.com",
                         "https://relay.example")  # fmt: skip
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, body: str) -> bool:
        self.sent.append((to, subject, body))
        return True

    def last_link(self, path: str) -> str:
        found = re.search(rf"https://relay\.example({re.escape(path)}\?t=[\w-]+)", self.sent[-1][2])
        assert found, self.sent[-1][2]
        return found.group(1)


def _site(tmp_path: Path, mail: bool = True) -> tuple[TestClient, UserStore, FakeMailer | None]:
    store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
    spaces = UserSpaces(store, Vault.from_env({"GPURELAY_VAULT_KEY": new_key()}), tmp_path / "u")
    mailer = FakeMailer() if mail else None
    settings = GatewaySettings(store=store, signer=SessionSigner(b"k" * 32), spaces=spaces,
                               mailer=mailer, secure_cookie=False)  # fmt: skip
    return TestClient(create_gateway(settings), follow_redirects=False), store, mailer


def _signup(client: TestClient, email: str = "ada@example.com") -> None:
    client.post(
        "/signup",
        data={"username": email.split("@")[0], "email": email, "password": PW, "confirm": PW},
    )


def test_with_email_new_accounts_confirm_before_setup(tmp_path: Path) -> None:
    client, store, mailer = _site(tmp_path)
    assert mailer is not None
    _signup(client)
    assert mailer.sent[-1][0] == "ada@example.com"
    assert client.get("/api/me").json()["user"]["verified"] is False
    assert client.get("/setup").headers["location"] == "/account?e=verify"
    assert client.get("/api/setup").status_code == 403
    assert client.get("/app/").headers["location"] == "/account?e=verify"
    link = mailer.last_link("/verify")
    assert client.get(link).headers["location"] == "/account?m=verified"
    assert client.get("/api/setup").status_code == 200
    assert client.get(link).headers["location"] == "/account?e=link"  # works once
    raw = mailer.sent[-1][2].split("t=")[1].split()[0]
    assert store.take_link(raw, "verify") is None  # the database never holds the token itself


def test_without_email_no_verification_is_needed(tmp_path: Path) -> None:
    client, _, _ = _site(tmp_path, mail=False)
    _signup(client)
    assert client.get("/api/me").json()["user"]["verified"] is True
    assert client.get("/api/setup").status_code == 200
    assert client.post("/forgot", data={"email": "ada@example.com"}).headers[
        "location"] == "/forgot?e=nomail"  # fmt: skip


def test_forgot_password_answers_the_same_and_resets_once(tmp_path: Path) -> None:
    client, store, mailer = _site(tmp_path)
    assert mailer is not None
    _signup(client)
    other = TestClient(client.app, follow_redirects=False)
    other.post("/signin", data={"email": "ada@example.com", "password": PW})
    before = len(mailer.sent)
    unknown = client.post("/forgot", data={"email": "nobody@example.com"})
    assert unknown.headers["location"] == "/forgot?m=sent" and len(mailer.sent) == before
    known = client.post("/forgot", data={"email": "ADA@example.com"})
    assert known.headers["location"] == "/forgot?m=sent" and len(mailer.sent) == before + 1
    token = mailer.last_link("/reset").split("t=")[1]
    assert client.get(f"/reset?t={token}").status_code == 200  # the page does not use it up
    short = client.post("/reset", data={"t": token, "password": "short", "confirm": "short"})
    assert "e=password" in short.headers["location"]
    done = client.post("/reset", data={"t": token, "password": NEW, "confirm": NEW})
    assert done.headers["location"] == "/signin?m=reset"
    assert other.get("/api/me").json()["user"] is None  # every session ended
    assert store.authenticate("ada@example.com", NEW) is not None
    again = client.post("/reset", data={"t": token, "password": PW, "confirm": PW})
    assert again.headers["location"] == "/forgot?e=link"


def test_reset_links_expire(tmp_path: Path) -> None:
    _, store, _ = _site(tmp_path)
    user = store.create("ada@example.com", PW)
    store.put_link(token_hash("tok"), user.id, "reset", store.clock() - 1)
    assert store.take_link(token_hash("tok"), "reset") is None


def test_change_password_keeps_this_device_signed_in_only(tmp_path: Path) -> None:
    client, _store, _ = _site(tmp_path, mail=False)
    _signup(client)
    phone = TestClient(client.app, follow_redirects=False)
    phone.post("/signin", data={"email": "ada@example.com", "password": PW})
    wrong = client.post("/api/account/password", json={"current": "nope nope nope", "new": NEW})
    assert wrong.status_code == 400
    assert (
        client.post("/api/account/password", json={"current": PW, "new": "short"}).status_code
        == 400
    )
    assert client.post("/api/account/password", json={"current": PW, "new": NEW}).json()["ok"]
    assert client.get("/api/me").json()["user"]["email"] == "ada@example.com"
    assert phone.get("/api/me").json()["user"] is None


def test_notices_toggle_and_account_view(tmp_path: Path) -> None:
    client, _store, _ = _site(tmp_path)
    _signup(client)
    assert client.get("/api/account").json() == {"email": "ada@example.com", "username": "ada",
                                                 "verified": False,
                                                 "notify": True, "mail": True}  # fmt: skip
    client.post("/api/account/notify", json={"notify": False})
    assert client.get("/api/account").json()["notify"] is False


def test_delete_account_removes_everything_on_this_server(tmp_path: Path) -> None:
    client, store, _ = _site(tmp_path, mail=False)
    _signup(client)
    user = store.find("ada@example.com")
    assert user is not None
    store.put_secret(user.id, "hf_token", "v1:sealed")
    (tmp_path / "u" / user.id).mkdir(parents=True)
    (tmp_path / "u" / user.id / "state.db").write_text("x")
    assert (
        client.post("/api/account/delete", json={"password": "wrong wrong wrong"}).status_code
        == 400
    )
    assert client.post("/api/account/delete", json={"password": PW}).json()["ok"]
    assert store.find("ada@example.com") is None and store.secret_names(user.id) == {}
    assert not (tmp_path / "u" / user.id).exists()
    assert client.get("/api/me").json()["user"] is None
    _signup(client)  # the email can be used again
    assert store.find("ada@example.com") is not None


def test_job_emails_on_changes_only(tmp_path: Path) -> None:
    store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
    mailer = FakeMailer()
    notices = Notices(store, mailer)
    user = store.create("ada@example.com", PW)
    store.set_email_verified(user.id)
    jobs = [{"id": "j1", "name": "train", "state": "running"},
            {"id": "j2", "name": "eval", "state": "completed"}]  # fmt: skip
    assert notices(user.id, jobs) == []  # the first look only learns the states
    jobs[0]["state"] = "awaiting_user"
    assert notices(user.id, jobs) == ["GPU Relay: train needs you on Colab"]
    assert notices(user.id, jobs) == []  # nothing new
    assert "/app/#/job/j1" in mailer.sent[-1][2]
    jobs[0]["state"] = "failed_infra"
    store.set_notify(user.id, False)
    assert notices(user.id, jobs) == []  # turned off
    store.set_notify(user.id, True)
    jobs[0]["state"] = "completed"
    assert notices(user.id, jobs) == ["GPU Relay: train finished"]


def test_mail_settings_from_the_environment() -> None:
    assert Mailer.from_env({}) is None
    with pytest.raises(ConfigError):
        Mailer.from_env({"GPURELAY_SMTP_HOST": "smtp.example"})  # no site address
    with pytest.raises(ConfigError):
        Mailer.from_env({"GPURELAY_SMTP_HOST": "smtp.example", "GPURELAY_SITE_URL": "https://x"})
    mailer = Mailer.from_env({"GPURELAY_SMTP_HOST": "smtp.example", "GPURELAY_SMTP_USER": "me",
                              "GPURELAY_SITE_URL": "https://x.example/"})  # fmt: skip
    assert mailer is not None and mailer.sender == "me" and mailer.port == 587
    assert mailer.link("/verify") == "https://x.example/verify"


def test_pages_exist(tmp_path: Path) -> None:
    client, _, _ = _site(tmp_path)
    for path in ("/forgot", "/reset?t=x"):
        assert client.get(path).status_code == 200
    _signup(client)
    assert "Delete your account" in client.get("/account").text


def test_choose_a_username_on_the_account_page(tmp_path: Path) -> None:
    client, store, _ = _site(tmp_path, mail=False)
    store.create("old@example.com", PW)  # an account from before usernames
    client.post("/signin", data={"login": "old@example.com", "password": PW})
    assert client.get("/api/account").json()["username"] is None
    bad = client.post("/api/account/username", json={"username": "x"})
    assert bad.status_code == 400 and bad.json()["code"] == "username"
    assert client.post("/api/account/username", json={"username": "Oldie"}).json() == {
        "ok": True, "username": "oldie"}  # fmt: skip
    other = TestClient(client.app, follow_redirects=False)
    assert other.post("/signin", data={"login": "oldie", "password": PW}).status_code == 303
    assert other.get("/api/me").json()["user"]["name"] == "oldie"
