from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from gpurelay.accounts.sessions import SessionSigner
from gpurelay.accounts.store import UserStore
from gpurelay.accounts.vault import Vault, new_key
from gpurelay.dashboard.desktop_api import LAUNCH_COOKIE
from gpurelay.errors import GpuRelayError
from gpurelay.gateway import children as children_mod
from gpurelay.gateway.app import GatewaySettings, create_gateway
from gpurelay.gateway.children import Children, SiteFull
from gpurelay.gateway.space import UserSpaces

PW = "correct horse battery"


class FakeProcess:
    def __init__(self) -> None:
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code

    def wait(self, timeout: float | None = None) -> int:
        self.code = 0
        return 0

    def kill(self) -> None:
        self.code = -9


class World:
    """Users, their settings, and fake GPU Relay processes with scripted job lists."""

    def __init__(self, tmp_path: Path, max_active: int = 2) -> None:
        self.store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
        self.spaces = UserSpaces(self.store, Vault.from_env({"GPURELAY_VAULT_KEY": new_key()}),
                                 tmp_path / "users")  # fmt: skip
        self.now = 1000.0
        self.spawned: list[tuple[Path, dict[str, str], int]] = []
        self.processes: list[FakeProcess] = []
        self.jobs: dict[int, list[dict[str, Any]]] = {}  # port -> jobs
        self.children = Children(self.spaces, max_active=max_active, idle_s=600,
                                 spawn=self.spawn, get_json=self.get_json,
                                 clock=lambda: self.now)  # fmt: skip

    def spawn(self, home: Path, env: dict[str, str], port: int) -> FakeProcess:
        self.spawned.append((home, env, port))
        process = FakeProcess()
        self.processes.append(process)
        return process

    def get_json(self, url: str, key: str) -> Any:
        port = int(url.split(":")[2].split("/")[0])
        if url.endswith("/healthz"):
            return {"ok": True}
        return {"jobs": self.jobs.get(port, [])}

    def user(self, email: str, storage: bool = True) -> str:
        user = self.store.create(email, PW)
        if storage:
            self.spaces.put_secrets(user.id, {"hf_token": "hf_" + "x" * 30})
            config = self.spaces.config(user.id)
            config.storage.hf_repo_id = f"{email.split('@')[0]}/gpurelay-store"
            self.spaces.save_config(user.id, config)
        return user.id


def test_a_users_process_gets_only_their_tokens_and_starts_once(tmp_path: Path) -> None:
    w = World(tmp_path)
    ada = w.user("ada@example.com")
    first = w.children.ensure(ada)
    assert w.children.ensure(ada) is first and len(w.spawned) == 1
    home, env, _ = w.spawned[0]
    assert home == tmp_path / "users" / ada and (home / "config.toml").exists()
    assert env["HF_TOKEN"] == "hf_" + "x" * 30 and env["GPURELAY_HOME"] == str(home)
    assert env["GPURELAY_LAUNCH_KEY"] == first.key and "GPURELAY_VAULT_KEY" not in env
    assert "x" * 30 not in (home / "config.toml").read_text(encoding="utf-8")


def test_no_process_without_storage(tmp_path: Path) -> None:
    w = World(tmp_path)
    bob = w.user("bob@example.com", storage=False)
    assert not w.children.can_start(bob)
    with pytest.raises(GpuRelayError):
        w.children.ensure(bob)


def test_a_full_site_stops_an_idle_user_never_an_active_one(tmp_path: Path) -> None:
    w = World(tmp_path, max_active=2)
    ada, bob, cy = w.user("ada@example.com"), w.user("bob@example.com"), w.user("cy@example.com")
    a = w.children.ensure(ada)
    w.now += 10
    b = w.children.ensure(bob)
    w.jobs[b.port] = [{"state": "running"}]
    w.children.tick()
    assert b.active and w.store.get(bob).active  # remembered for the next site start
    w.children.ensure(cy)  # room made by stopping ada (idle), not bob (a job in flight)
    assert w.children.get(ada) is None and (a.home / "server.stop").exists()
    assert w.children.get(bob) is b
    w.jobs[w.children.get(cy).port] = [{"state": "queued"}]  # type: ignore[union-attr]
    w.children.tick()
    with pytest.raises(SiteFull):
        w.children.ensure(ada)


def test_idle_users_stop_and_finished_jobs_count_as_idle(tmp_path: Path) -> None:
    w = World(tmp_path)
    ada = w.user("ada@example.com")
    child = w.children.ensure(ada)
    w.jobs[child.port] = [{"state": "running"}]
    w.now += 3600
    w.children.tick()
    assert w.children.get(ada) is child  # a job in flight: never stopped for being idle
    w.jobs[child.port] = [{"state": "completed"}, {"state": "paused"}]
    w.children.tick()
    assert w.children.get(ada) is None and not w.store.get(ada).active


def test_a_crashed_process_with_jobs_in_flight_starts_again(tmp_path: Path) -> None:
    w = World(tmp_path)
    ada = w.user("ada@example.com")
    child = w.children.ensure(ada)
    w.jobs[child.port] = [{"state": "awaiting_user"}]
    w.children.tick()
    w.processes[0].code = 1
    w.children.tick()
    assert len(w.spawned) == 2 and w.children.get(ada) is not None


def test_site_start_brings_back_users_with_jobs_in_flight(tmp_path: Path) -> None:
    w = World(tmp_path)
    ada, bob = w.user("ada@example.com"), w.user("bob@example.com")
    w.store.set_active(ada, True)
    w.children.start_active()
    assert w.children.get(ada) is not None and w.children.get(bob) is None


def test_restart_applies_new_settings_only_to_a_running_process(tmp_path: Path) -> None:
    w = World(tmp_path)
    ada = w.user("ada@example.com")
    w.children.restart(ada)
    assert w.spawned == []
    old = w.children.ensure(ada)
    w.children.restart(ada)
    assert len(w.spawned) == 2 and w.children.get(ada) is not old


def test_stop_all_asks_every_process_to_stop(tmp_path: Path) -> None:
    w = World(tmp_path)
    homes = [w.children.ensure(w.user(e)).home for e in ("a@example.com", "b@example.com")]
    w.children.stop_all()
    assert all((h / "server.stop").exists() for h in homes) and w.children.running() == []


# ---------------------------------------------------------------- forwarding


def _fake_user_app(seen: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def echo(path: str, request: Request) -> Any:
        seen.append({"path": path, "cookies": dict(request.cookies), "method": request.method,
                     "body": (await request.body()).decode(), "query": str(request.query_params),
                     "action": request.headers.get("x-gpurelay-action")})  # fmt: skip
        from fastapi.responses import JSONResponse

        response = JSONResponse({"path": path})
        response.set_cookie("child_cookie", "x")
        return response

    return app


@pytest.fixture
def forwarding(tmp_path: Path) -> Any:
    w = World(tmp_path)
    seen: list[dict[str, Any]] = []
    transport = httpx.ASGITransport(app=_fake_user_app(seen))

    def client(timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, timeout=timeout)

    settings = GatewaySettings(store=w.store, signer=SessionSigner(b"k" * 32), spaces=w.spaces,
                               children=w.children, secure_cookie=False,
                               proxy_client=client)  # fmt: skip
    app = create_gateway(settings)
    return w, TestClient(app, follow_redirects=False), seen


def _sign_in(client: TestClient, email: str) -> None:
    client.post("/signin", data={"email": email, "password": PW})


def test_app_forwards_to_the_signed_in_users_process_only(forwarding: Any) -> None:
    w, client, seen = forwarding
    ada = w.user("ada@example.com")
    assert client.get("/app/").status_code == 303  # not signed in: to /signin
    assert client.get("/app/api/overview").status_code == 401  # the app's own calls: 401
    _sign_in(client, "ada@example.com")
    client.cookies.set("gpurelay_user_extra", "browser-cookie")
    response = client.post(
        "/app/api/jobs/j1/cancel?x=1",
        content="{}",
        headers={"x-gpurelay-action": "1", "content-type": "application/json"},
    )
    assert response.json() == {"path": "api/jobs/j1/cancel"}
    assert "set-cookie" not in response.headers  # the user process's cookies stay inside
    [request] = seen
    child = w.children.get(ada)
    assert request["cookies"] == {LAUNCH_COOKIE: child.key}  # browser cookies not passed on
    assert request["query"] == "x=1" and request["action"] == "1" and request["body"] == "{}"


def test_app_without_storage_goes_to_setup(forwarding: Any) -> None:
    w, client, seen = forwarding
    w.user("bob@example.com", storage=False)
    _sign_in(client, "bob@example.com")
    assert client.get("/app/").headers["location"] == "/setup"
    assert client.get("/app/api/overview").status_code == 409 and seen == []


def test_each_user_reaches_their_own_process(forwarding: Any) -> None:
    w, client, seen = forwarding
    ada, bob = w.user("ada@example.com"), w.user("bob@example.com")
    _sign_in(client, "ada@example.com")
    client.get("/app/api/overview")
    other = TestClient(client.app, follow_redirects=False)
    _sign_in(other, "bob@example.com")
    other.get("/app/api/overview")
    keys = [r["cookies"][LAUNCH_COOKIE] for r in seen]
    assert keys == [w.children.get(ada).key, w.children.get(bob).key] and keys[0] != keys[1]


def test_unfinished_jobs_per_user_are_limited(forwarding: Any) -> None:
    w, client, _seen = forwarding
    ada = w.user("ada@example.com")
    _sign_in(client, "ada@example.com")
    client.get("/app/api/overview")
    w.jobs[w.children.get(ada).port] = [
        {"state": "running"},
        {"state": "queued"},
        {"state": "awaiting_user"},
        {"state": "completed"},
    ]
    refused = client.post("/app/api/submit", content="{}")
    assert refused.status_code == 429 and "3 unfinished jobs" in refused.json()["detail"]
    w.jobs[w.children.get(ada).port] = [{"state": "running"}, {"state": "cancelled"}]
    assert client.post("/app/api/submit", content="{}").status_code == 200


def test_free_port_is_usable() -> None:
    assert 0 < children_mod.free_port() < 65536
