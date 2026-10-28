"""The AI helper's way in (spec §20.2, plan A1).

A user's AI tools (Claude Code under their own plan) are on their laptop, which the website
cannot reach. ``gpurelay helper`` runs there and asks the website for work instead, with a
**helper key** the user makes on /setup (shown once; only its hash is kept; a new key replaces
the old one, and removing it locks the helper out):

* ``POST /helper/v1/poll``: "I am here, with these tools"; the answer is the next fix
  request to work on, if the user's GPU Relay is running and has one;
* ``POST /helper/v1/fixes/<id>``: the AI's answer to a request.

Both are passed to the user's own GPU Relay process, which keeps the requests and checks
every answer against the code it sent. Polls never start that process or keep it awake.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from gpurelay.accounts.store import UserStore
from gpurelay.aifix.board import clean_tools
from gpurelay.dashboard.auth import RateLimiter
from gpurelay.dashboard.desktop_api import LAUNCH_COOKIE
from gpurelay.errors import GpuRelayError
from gpurelay.gateway.account_routes import token_hash
from gpurelay.gateway.children import Children, SiteFull

HELPER_PREFIX = "/helper/"
KEY_KIND = "helper"
KEY_PREFIX = "grh_"
KEY_LIFETIME_S = 10 * 365 * 24 * 3600.0  # until replaced or removed
MAX_BODY = 2 * 1024 * 1024
SEEN_S = 30.0  # a helper that asked within this long is connected


def helper_router(
    store: UserStore,
    children: Children,
    client_factory: Any = None,
    limiter: RateLimiter | None = None,
    clock: Callable[[], float] | None = None,
) -> APIRouter:
    router = APIRouter()
    limiter = limiter or RateLimiter(per_client=10, total=100, window_s=900.0)
    now = clock or store.clock
    seen: dict[str, dict[str, Any]] = {}  # user id -> last poll and tools (for /setup)
    state: dict[str, Any] = {}

    def client() -> Any:
        if "client" not in state:
            import httpx

            state["client"] = (client_factory or httpx.AsyncClient)(timeout=60.0)
        return state["client"]

    # ---------------------------------------------------------------- the user, on /setup

    @router.get("/api/ai/helper")
    def helper_status(request: Request) -> dict[str, Any]:
        user_id = request.state.user.id
        info = seen.get(user_id, {})
        last = info.get("last_seen")
        return {
            "key": store.has_link(user_id, KEY_KIND),
            "connected": last is not None and now() - last <= SEEN_S,
            "last_seen": last,
            "tools": info.get("tools", []),
        }

    @router.post("/api/ai/helper/key")
    def new_key(request: Request) -> dict[str, Any]:
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        store.put_link(token_hash(key), request.state.user.id, KEY_KIND, now() + KEY_LIFETIME_S)
        return {"key": key}

    @router.post("/api/ai/helper/remove")
    def remove_key(request: Request) -> dict[str, Any]:
        store.delete_links(request.state.user.id, KEY_KIND)
        seen.pop(request.state.user.id, None)
        return {"ok": True}

    # ---------------------------------------------------------------- the helper

    def client_id(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        return forwarded or (request.client.host if request.client else "?")

    def helper_user(request: Request) -> str | None:
        auth = request.headers.get("authorization", "")
        key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        if not key.startswith(KEY_PREFIX):
            return None
        return store.link_user(token_hash(key), KEY_KIND)

    async def forward(user_id: str, path: str, body: bytes, start: bool) -> Any:
        child = children.get(user_id)
        if child is None and start:
            try:
                child = await run_in_threadpool(children.ensure, user_id)
            except (SiteFull, GpuRelayError) as exc:
                return JSONResponse({"detail": str(exc)}, status_code=503)
        if child is None:
            return None
        headers = {"cookie": f"{LAUNCH_COOKIE}={child.key}", "x-gpurelay-action": "1",
                   "content-type": "application/json"}  # fmt: skip
        try:
            upstream = await client().post(f"{child.url}/{path}", content=body, headers=headers)
        except Exception:
            return JSONResponse({"detail": "your GPU Relay did not answer; try again"}, 502)
        try:
            data = upstream.json()
        except ValueError:
            data = {"detail": "unexpected answer"}
        return JSONResponse(data, status_code=upstream.status_code)

    async def authorised(request: Request) -> tuple[str | None, bytes, Any]:
        client = client_id(request)
        if limiter.blocked(client):
            return None, b"", JSONResponse({"detail": "too many wrong keys; wait"}, 429)
        user_id = await run_in_threadpool(helper_user, request)
        if user_id is None:
            limiter.failed(client)
            return (
                None,
                b"",
                JSONResponse(
                    {"detail": "unknown helper key: make a new one on the setup page"}, 401
                ),
            )
        body = await request.body()
        if len(body) > MAX_BODY:
            return None, b"", JSONResponse({"detail": "the answer is too large"}, 413)
        return user_id, body, None

    @router.post("/helper/v1/poll")
    async def poll(request: Request) -> Any:
        user_id, body, refused = await authorised(request)
        if refused is not None:
            return refused
        assert user_id is not None
        import json

        try:
            tools = json.loads(body or b"{}").get("tools", [])
        except (ValueError, AttributeError):
            tools = []
        seen[user_id] = {"last_seen": now(), "tools": clean_tools(tools)}
        answer = await forward(user_id, "api/helper/poll", body, start=False)
        # Not running: nobody can have asked for a fix since it stopped.
        return answer if answer is not None else {"fix": None, "app": "stopped"}

    @router.post("/helper/v1/fixes/{fix_id}")
    async def answer(fix_id: str, request: Request) -> Any:
        user_id, body, refused = await authorised(request)
        if refused is not None:
            return refused
        assert user_id is not None
        if not fix_id.startswith("fix-") or not fix_id[4:].isalnum():
            return JSONResponse({"detail": "no such request"}, 404)
        return await forward(user_id, f"api/helper/fixes/{fix_id}", body, start=True)

    return router
