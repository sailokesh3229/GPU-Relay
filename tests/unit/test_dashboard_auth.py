"""Dashboard login for `gpurelay serve` (plan W1): roles, sessions, rate limits, headers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from test_controller import Harness, _harness

from gpurelay.controller.service import Context
from gpurelay.dashboard.app import ACTION_HEADER, create_app
from gpurelay.dashboard.auth import COOKIE_NAME, OWNER, VIEWER, Auth, RateLimiter
from gpurelay.errors import ConfigError
from gpurelay.models import JobState

ACT = {ACTION_HEADER: "1"}
OWNER_PW = "owner-password-123"
VIEWER_PW = "viewer-password-456"


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def _auth(clock: Clock | None = None, **kw: object) -> Auth:
    clock = clock or Clock()
    return Auth(OWNER_PW, VIEWER_PW, session_key=b"k" * 32, clock=clock,
                limiter=RateLimiter(clock=clock), **kw)  # type: ignore[arg-type]  # fmt: skip


def _client(tmp_path: Path, h: Harness, auth: Auth) -> TestClient:
    ctx = Context(h.controller.config, tmp_path / "home", h.db, h.storage, h.controller)
    app = create_app(ctx, clock=h.clock, allowed_hosts=None, auth=auth)
    return TestClient(app, base_url="https://gpu-relay.hf.space", follow_redirects=False)


def _login(client: TestClient, password: str) -> object:
    return client.post("/login", content=f"password={password}",
                       headers={"content-type": "application/x-www-form-urlencoded"})  # fmt: skip


@pytest.fixture
def setup(tmp_path: Path) -> tuple[TestClient, Harness]:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    return _client(tmp_path, h, _auth()), h


def test_everything_but_login_and_health_needs_a_session(setup: tuple[TestClient, Harness]) -> None:
    client, _ = setup
    page = client.get("/")
    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert client.get("/api/overview").status_code == 401
    assert client.post("/api/jobs/x/pause", headers=ACT).status_code == 401
    assert client.get("/healthz").json() == {"ok": True}
    login = client.get("/login")
    assert login.status_code == 200 and 'type="password"' in login.text
    assert "https://" not in login.text and "innerHTML" not in login.text


def test_owner_can_act_viewer_can_only_read(setup: tuple[TestClient, Harness]) -> None:
    client, h = setup
    job = h.submit()
    h.controller.tick()

    r = _login(client, VIEWER_PW)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert client.get("/api/me").json() == {
        "role": VIEWER,
        "login": True,
        "anonymous": False,
        "desktop": False,
        "configured": True,
        "platforms_url": None,
    }
    assert client.get("/api/overview").status_code == 200
    assert client.get(f"/api/jobs/{job.id}").status_code == 200
    assert client.post(f"/api/jobs/{job.id}/pause", headers=ACT).status_code == 403
    assert h.job(job.id).requested is None

    _login(client, OWNER_PW)
    assert client.get("/api/me").json()["role"] == OWNER
    assert client.post(f"/api/jobs/{job.id}/pause", headers=ACT).status_code == 200
    assert h.job(job.id).requested == "pause"
    assert h.job(job.id).state is JobState.RUNNING  # applied by the controller later


def test_session_cookie_flags_and_logout(setup: tuple[TestClient, Harness]) -> None:
    client, _ = setup
    r = _login(client, OWNER_PW)
    cookie = r.headers["set-cookie"].lower()
    for flag in ("httponly", "secure", "samesite=lax", "path=/"):
        assert flag in cookie
    assert client.get("/api/overview").status_code == 200
    out = client.post("/logout")
    assert out.status_code == 303 and "max-age=0" in out.headers["set-cookie"].lower()
    client.cookies.clear()
    assert client.get("/api/overview").status_code == 401


def test_wrong_password_and_forged_or_expired_sessions(tmp_path: Path) -> None:
    clock = Clock()
    auth = _auth(clock)
    client = _client(tmp_path, _harness(tmp_path, {"a": (None, True, 1, "T4")}), auth)
    r = _login(client, "not-the-password")
    assert r.headers["location"] == "/login?e=wrong" and COOKIE_NAME not in client.cookies

    forged = auth.issue(OWNER)[:-2] + "AA"
    client.cookies.set(COOKIE_NAME, forged)
    assert client.get("/api/overview").status_code == 401
    other_key = Auth(OWNER_PW, VIEWER_PW, session_key=b"x" * 32, clock=clock)
    client.cookies.set(COOKIE_NAME, other_key.issue(OWNER))
    assert client.get("/api/overview").status_code == 401

    client.cookies.set(COOKIE_NAME, auth.issue(OWNER))
    assert client.get("/api/overview").status_code == 200
    clock.t += auth.session_s + 1
    assert client.get("/api/overview").status_code == 401


def test_changing_a_password_ends_existing_sessions() -> None:
    old = Auth(OWNER_PW, VIEWER_PW, session_key=b"k" * 32)
    token = old.issue(OWNER)
    assert old.verify(token) == OWNER
    assert Auth("a-new-owner-password", VIEWER_PW, session_key=b"k" * 32).verify(token) is None
    viewer_token = old.issue(VIEWER)
    assert Auth(OWNER_PW, None, session_key=b"k" * 32).verify(viewer_token) is None


def test_login_is_rate_limited_per_client_and_in_total(tmp_path: Path) -> None:
    clock = Clock()
    client = _client(tmp_path, _harness(tmp_path, {"a": (None, True, 1, "T4")}), _auth(clock))
    ip = {"x-forwarded-for": "203.0.113.7"}
    for _ in range(5):
        client.post("/login", content="password=nope", headers=ip)
    r = client.post("/login", content=f"password={OWNER_PW}", headers=ip)
    assert r.headers["location"] == "/login?e=busy"  # even the right password, for a while
    other = client.post("/login", content=f"password={OWNER_PW}",
                        headers={"x-forwarded-for": "198.51.100.1"})  # fmt: skip
    assert other.headers["location"] == "/"
    clock.t += 901
    assert (
        client.post("/login", content=f"password={OWNER_PW}", headers=ip).headers["location"] == "/"
    )

    limiter = RateLimiter(per_client=100, total=3, clock=clock)
    for i in range(3):
        limiter.failed(f"10.0.0.{i}")
    assert limiter.blocked("10.9.9.9")  # spread over many addresses: still capped


def test_hosted_responses_carry_security_headers(setup: tuple[TestClient, Harness]) -> None:
    client, _ = setup
    for r in (client.get("/login"), client.get("/api/overview")):
        csp = r.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in csp
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["cache-control"] == "no-store"


def test_password_rules_and_env() -> None:
    with pytest.raises(ConfigError, match="at least 12"):
        Auth("short")
    with pytest.raises(ConfigError, match="viewer password must have"):
        Auth(OWNER_PW, "short")
    with pytest.raises(ConfigError, match="differ"):
        Auth(OWNER_PW, OWNER_PW)
    with pytest.raises(ConfigError, match="GPURELAY_OWNER_PASSWORD"):
        Auth.from_env({})
    auth = Auth.from_env({"GPURELAY_OWNER_PASSWORD": OWNER_PW, "GPURELAY_VIEWER_PASSWORD": ""})
    assert auth.viewer_password is None and auth.check_password(OWNER_PW) == OWNER
    assert auth.check_password("") is None


def test_any_host_is_only_allowed_with_a_login(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    ctx = Context(h.controller.config, tmp_path / "home", h.db, h.storage, h.controller)
    with pytest.raises(ValueError, match="needs a login"):
        create_app(ctx, allowed_hosts=None)


def test_public_view_lets_guests_look_but_never_act(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    auth = Auth(OWNER_PW, None, session_key=b"k" * 32, public_view=True)
    client = _client(tmp_path, h, auth)
    job = h.submit()
    page = client.get("/")
    assert page.status_code == 200 and "<title>GPU Relay</title>" in page.text
    assert client.get("/api/me").json() == {
        "role": VIEWER,
        "login": True,
        "anonymous": True,
        "desktop": False,
        "configured": True,
        "platforms_url": None,
    }
    assert client.get("/api/overview").status_code == 200
    assert client.get(f"/api/jobs/{job.id}").status_code == 200
    assert client.get("/api/settings").json()["public_view"] is True
    for url in (f"/api/jobs/{job.id}/pause", "/api/submit", "/api/analyze", "/api/quota/a"):
        assert client.post(url, json={}, headers=ACT).status_code == 403
    assert h.job(job.id).requested is None
    assert client.get("/login").status_code == 200  # the owner can still log in

    _login(client, OWNER_PW)
    assert client.get("/api/me").json() == {
        "role": OWNER,
        "login": True,
        "anonymous": False,
        "desktop": False,
        "configured": True,
        "platforms_url": None,
    }
    assert client.post(f"/api/jobs/{job.id}/pause", headers=ACT).status_code == 200


def test_public_view_is_off_unless_asked_for() -> None:
    env = {"GPURELAY_OWNER_PASSWORD": OWNER_PW}
    assert Auth.from_env(env).public_view is False
    assert Auth.from_env({**env, "GPURELAY_PUBLIC_VIEW": "1"}).public_view is True
    assert Auth.from_env({**env, "GPURELAY_PUBLIC_VIEW": "no"}).public_view is False
