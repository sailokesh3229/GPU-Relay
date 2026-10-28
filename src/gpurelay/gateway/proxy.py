"""``/app/...``: a signed-in user's own GPU Relay, forwarded to their process (plan M3).

The user's web app is the same one as on the desktop and uses relative links, so it works
under ``/app/``. Only the signed-in user's process is ever reached; the launch key is added
here and never shown to the browser. Cookies from the browser are not passed on, and the
user process's own cookies are not passed back.
"""

from __future__ import annotations

import html
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse, Response

from gpurelay.dashboard.desktop_api import LAUNCH_COOKIE
from gpurelay.dashboard.forms import MAX_BODY_BYTES
from gpurelay.errors import GpuRelayError
from gpurelay.gateway.children import Children, SiteFull

FORWARD_REQUEST = ("content-type", "accept", "x-gpurelay-action", "range")
FORWARD_RESPONSE = ("content-type", "content-disposition", "content-range", "accept-ranges")
METHODS = ["GET", "POST", "PUT", "DELETE"]

# A slim bar above the user's GPU Relay page: back to setup and the account without the
# browser's back button. The app page is the same one the desktop app shows, so the bar is
# added here, on the website only.
SITE_BAR = (
    '<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:8px 16px;'
    'background:var(--card,#fff);border-bottom:1px solid var(--line,#e2e5ea);font-size:13px">'
    '<a href="/setup" style="font-weight:600">Set up accounts</a>'
    '<a href="/account">Account</a>'
    '<span style="margin-left:auto;color:var(--muted,#5b6270)">Signed in as {name}</span>'
    '<form method="post" action="/signout" style="margin:0">'
    '<button type="submit" style="min-height:28px;padding:0 10px;border-radius:6px;'
    'border:1px solid var(--line,#e2e5ea);background:transparent;cursor:pointer">'
    "Sign out</button></form></div>"
)


def with_site_bar(page: bytes, name: str) -> bytes:
    bar = SITE_BAR.replace("{name}", html.escape(name)).encode()
    head, found, rest = page.partition(b"<body>")
    return head + found + bar + rest if found else page


def app_router(children: Children, client_factory: Any = None, max_jobs: int = 3) -> APIRouter:
    """``max_jobs``: unfinished jobs a user may have at once (the site's capacity is small)."""
    router = APIRouter()
    state: dict[str, Any] = {}

    def client() -> Any:
        if "client" not in state:
            import httpx

            state["client"] = (client_factory or httpx.AsyncClient)(timeout=120.0)
        return state["client"]

    @router.get("/app")
    def app_root() -> Any:
        return RedirectResponse("/app/", status_code=303)

    @router.api_route("/app/{path:path}", methods=METHODS)
    async def forward(path: str, request: Request) -> Any:
        user_id = str(request.state.user.id)
        wants_page = request.method == "GET" and not path.startswith("api/")
        if not await run_in_threadpool(children.can_start, user_id):
            if wants_page:
                return RedirectResponse("/setup", status_code=303)
            return JSONResponse({"detail": "connect Hugging Face storage first"}, 409)
        try:
            child = await run_in_threadpool(children.ensure, user_id)
        except SiteFull as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except GpuRelayError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=502)
        if request.method == "POST" and path == "api/submit":
            try:
                in_flight = await run_in_threadpool(children.jobs_in_flight, user_id)
            except Exception:
                in_flight = 0  # the user's GPU Relay decides; it is busy, not over the limit
            if in_flight >= max_jobs:
                return JSONResponse({"detail": f"you have {in_flight} unfinished jobs; this site "
                                     f"allows {max_jobs} at a time"}, status_code=429)  # fmt: skip
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "the upload is too large"}, status_code=413)
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQUEST}
        headers["cookie"] = f"{LAUNCH_COOKIE}={child.key}"
        try:
            upstream = await client().request(
                request.method, f"{child.url}/{path}", params=list(request.query_params.items()),
                content=body, headers=headers,
            )  # fmt: skip
        except Exception:
            return JSONResponse({"detail": "your GPU Relay did not answer; try again"}, 502)
        kept = {k: v for k, v in upstream.headers.items() if k.lower() in FORWARD_RESPONSE}
        content = upstream.content
        if path == "" and upstream.headers.get("content-type", "").startswith("text/html"):
            content = with_site_bar(content, request.state.user.name)
        return Response(content, status_code=upstream.status_code, headers=kept)

    return router
