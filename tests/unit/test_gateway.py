from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from gpurelay.accounts.sessions import COOKIE_NAME, SessionSigner
from gpurelay.accounts.store import UserStore
from gpurelay.accounts.vault import new_key
from gpurelay.errors import ConfigError
from gpurelay.gateway.app import GatewaySettings, create_gateway

PW = "correct horse battery"


def _settings(tmp_path: Path, **kw: Any) -> GatewaySettings:
    store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
    return GatewaySettings(store=store, signer=SessionSigner(b"k" * 32), secure_cookie=False, **kw)


def _client(settings: GatewaySettings) -> TestClient:
    return TestClient(create_gateway(settings), follow_redirects=False)


def _signup(client: TestClient, email: str, password: str = PW, **extra: str) -> Any:
    confirm = extra.pop("confirm", password)
    data = {"email": email, "password": password, "confirm": confirm, **extra}
    return client.post("/signup", data=data)


def test_sign_up_signs_in_and_shows_the_account(tmp_path: Path) -> None:
    client = _client(_settings(tmp_path))
    assert client.get("/api/me").json()["user"] is None
    response = _signup(client, "ada@example.com")
    assert response.status_code == 303 and response.headers["location"] == "/"
    assert "httponly" in response.headers["set-cookie"].lower()
    assert client.get("/api/me").json()["user"] == {"email": "ada@example.com", "verified": True}
    client.post("/signout")
    assert client.get("/api/me").json()["user"] is None


def test_sign_in_with_the_right_password_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.store.create("ada@example.com", PW)
    client = _client(settings)
    wrong = client.post("/signin", data={"email": "ada@example.com", "password": "nope nope nope"})
    assert wrong.headers["location"].startswith("/signin?e=wrong")
    assert COOKIE_NAME not in client.cookies
    client.post("/signin", data={"email": "ADA@example.com", "password": PW})
    assert client.get("/api/me").json()["user"]["email"] == "ada@example.com"


def test_failed_sign_ins_are_rate_limited(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.store.create("ada@example.com", PW)
    client = _client(settings)
    for _ in range(5):
        client.post("/signin", data={"email": "ada@example.com", "password": "guess guess guess"})
    blocked = client.post("/signin", data={"email": "ada@example.com", "password": PW})
    assert "e=busy" in blocked.headers["location"]


def test_sign_up_limits_invite_code_and_mismatch(tmp_path: Path) -> None:
    client = _client(_settings(tmp_path, max_users=1, invite_code="let-me-in"))
    assert "e=invite" in _signup(client, "a@example.com", invite="wrong").headers["location"]
    mismatch = _signup(client, "a@example.com", invite="let-me-in", confirm="something else!")
    assert "e=mismatch" in mismatch.headers["location"]
    assert _signup(client, "a@example.com", invite="let-me-in").headers["location"] == "/"
    other = _client(_settings(tmp_path, max_users=1, invite_code="let-me-in"))
    assert "e=full" in _signup(other, "b@example.com", invite="let-me-in").headers["location"]
    assert other.get("/api/me").json()["signup"] == {"open": False, "invite": True}


def test_errors_are_codes_never_free_text(tmp_path: Path) -> None:
    client = _client(_settings(tmp_path))
    _signup(client, "ada@example.com")
    again = _client(_settings(tmp_path))
    assert "e=exists" in _signup(again, "ada@example.com").headers["location"]
    assert "e=password" in _signup(again, "bob@example.com", "short").headers["location"]
    assert "e=email" in _signup(again, "bob", PW).headers["location"]


def test_sign_out_everywhere_ends_other_sessions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.store.create("ada@example.com", PW)
    laptop, phone = _client(settings), _client(settings)
    for c in (laptop, phone):
        c.post("/signin", data={"email": "ada@example.com", "password": PW})
    laptop.post("/signout-everywhere")
    assert phone.get("/api/me").json()["user"] is None


def test_cross_site_posts_are_refused(tmp_path: Path) -> None:
    client = _client(_settings(tmp_path))
    response = client.post("/signup", data={"email": "a@example.com", "password": PW,
                                            "confirm": PW},
                           headers={"origin": "https://evil.example"})  # fmt: skip
    assert response.status_code == 403
    assert client.get("/api/me").json()["user"] is None
    other_site = {"origin": "null", "referer": "https://evil.example/page"}
    null = client.post("/signin", data={"email": "a@example.com", "password": PW},
                       headers=other_site)  # fmt: skip
    assert null.status_code == 403  # "null" falls back to the Referer, which is another site


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://testserver"},  # Chrome, Edge, Firefox, Safari
        {"origin": "null"},  # strict privacy settings, no Referer
        {"origin": "null", "referer": "http://testserver/signup"},  # Brave-like
        {"referer": "http://testserver/signup"},  # older Safari and Firefox: no Origin
        {},  # tools, or a browser that sends neither
    ],
)
def test_every_browser_can_sign_up(tmp_path: Path, headers: dict[str, str]) -> None:
    client = _client(_settings(tmp_path))
    data = {"email": "ada@example.com", "password": PW, "confirm": PW}
    response = client.post("/signup", data=data, headers=headers)
    assert response.status_code == 303 and response.headers["location"] == "/"
    assert client.get("/api/me").json()["user"]["email"] == "ada@example.com"


def test_browsers_send_a_real_origin_on_this_sites_forms(tmp_path: Path) -> None:
    """A browser sends "Origin: null" on form posts under Referrer-Policy: no-referrer, which
    refused every sign-in (seen live); same-origin makes it send the site's origin."""
    client = _client(_settings(tmp_path))
    assert client.get("/signin").headers["referrer-policy"] == "same-origin"
    response = _signup(client, "ada@example.com")
    assert response.status_code == 303


def test_pages_and_headers(tmp_path: Path) -> None:
    client = _client(_settings(tmp_path))
    home = client.get("/")
    assert home.status_code == 200 and "Kaggle" in home.text and "write" in home.text
    assert "frame-ancestors 'none'" in home.headers["content-security-policy"]
    assert client.get("/signup").status_code == 200 and client.get("/signin").status_code == 200
    assert client.get("/healthz").json() == {"ok": True}
    assert client.post("/signout-everywhere").status_code == 303  # not signed in: to /signin
    assert client.get("/api/anything").status_code == 401


def test_settings_from_the_environment(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'u.db'}"
    with pytest.raises(ConfigError):
        GatewaySettings.from_env({"GPURELAY_SESSION_KEY": "k" * 32})
    with pytest.raises(ConfigError):
        GatewaySettings.from_env({"DATABASE_URL": url, "GPURELAY_SESSION_KEY": "short"})
    settings = GatewaySettings.from_env({
        "DATABASE_URL": url, "GPURELAY_SESSION_KEY": "k" * 32,
        "GPURELAY_MAX_USERS": "5", "GPURELAY_INVITE_CODE": " code ",
        "GPURELAY_VAULT_KEY": new_key(), "GPURELAY_USERS_DIR": str(tmp_path / "users"),
    })  # fmt: skip
    assert settings.max_users == 5 and settings.invite_code == "code"
    assert settings.spaces is not None and settings.spaces.root == tmp_path / "users"
