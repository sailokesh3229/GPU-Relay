"""The multi-user website (plan M1-M4): accounts in front of each user's own GPU Relay.

Each visitor signs up with an email and a password; nobody shares an account. A signed-in
user connects their platforms on /setup (tokens sealed in the vault, M2), and /app/ is their
own GPU Relay, a separate process with only their tokens (M3). Account care, email and job
notices are M4. Settings come from the environment (mail: :mod:`gpurelay.gateway.mail`):

* ``DATABASE_URL``: where accounts live (``postgresql://...``, or ``sqlite:///path``);
* ``GPURELAY_SESSION_KEY``: signs session cookies (16+ characters, stable across restarts);
* ``GPURELAY_MAX_USERS``: how many accounts may exist (default 3: the free plan's memory);
* ``GPURELAY_INVITE_CODE``: if set, sign-up needs it (share it with the people you invite).
* ``GPURELAY_VAULT_KEY``: encrypts users' platform tokens (``accounts/vault.py``);
* ``GPURELAY_USERS_DIR``: the users' folders on this server (default ``~/gpurelay-users``);
* ``GPURELAY_MAX_ACTIVE``: how many users' GPU Relay may run at once (default 2);
* ``GPURELAY_MAX_JOBS``: unfinished jobs per user (default 3).
"""

from __future__ import annotations

import contextlib
import hmac
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from gpurelay import __version__
from gpurelay.accounts.sessions import COOKIE_NAME, SessionSigner
from gpurelay.accounts.store import ENV_DATABASE_URL, AccountError, User, UserStore
from gpurelay.accounts.vault import Vault
from gpurelay.dashboard.auth import RateLimiter
from gpurelay.errors import ConfigError
from gpurelay.gateway.account_routes import account_router, send_verification
from gpurelay.gateway.children import Children
from gpurelay.gateway.mail import Mailer
from gpurelay.gateway.notices import Notices
from gpurelay.gateway.space import UserSpaces

ENV_SESSION_KEY = "GPURELAY_SESSION_KEY"
ENV_MAX_USERS = "GPURELAY_MAX_USERS"
ENV_INVITE_CODE = "GPURELAY_INVITE_CODE"
ENV_USERS_DIR = "GPURELAY_USERS_DIR"
ENV_MAX_ACTIVE = "GPURELAY_MAX_ACTIVE"
ENV_MAX_JOBS = "GPURELAY_MAX_JOBS"
TICK_S = 30.0

log = logging.getLogger(__name__)
DEFAULT_MAX_USERS = 3
MAX_FORM = 4096
PUBLIC_PATHS = frozenset(
    {"/", "/signin", "/signup", "/healthz", "/api/me", "/verify", "/forgot", "/reset"}
)
# With email on, these wait until the user confirmed their address.
NEEDS_VERIFIED = ("/setup", "/api/setup", "/app")
HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; "
    "form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@dataclass
class GatewaySettings:
    store: UserStore
    signer: SessionSigner
    spaces: UserSpaces | None = None  # tokens and settings (None: accounts only, for tests)
    children: Children | None = None  # each user's GPU Relay (None: no /app, for tests)
    proxy_client: Any = None  # makes the HTTP client for /app (tests pass a fake transport)
    mailer: Mailer | None = None  # None: no email (no verification, no password reset)
    max_jobs: int = 3
    max_users: int = DEFAULT_MAX_USERS
    invite_code: str | None = None
    secure_cookie: bool = True
    signin_limiter: RateLimiter = field(default_factory=RateLimiter)
    after_change: Callable[[str], None] = lambda user_id: None  # a user's settings changed
    signup_limiter: RateLimiter = field(
        default_factory=lambda: RateLimiter(per_client=5, total=30, window_s=3600.0)
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GatewaySettings:
        env = os.environ if env is None else env
        url = env.get(ENV_DATABASE_URL, "").strip()
        if not url:
            raise ConfigError(f"set {ENV_DATABASE_URL} (a Postgres URL): where accounts live")
        key = env.get(ENV_SESSION_KEY, "")
        if len(key) < 16:
            raise ConfigError(f"set {ENV_SESSION_KEY} to a random value of 16+ characters")
        try:
            max_users = int(env.get(ENV_MAX_USERS, "") or DEFAULT_MAX_USERS)
            max_active = int(env.get(ENV_MAX_ACTIVE, "") or 2)
            max_jobs = int(env.get(ENV_MAX_JOBS, "") or 3)
        except ValueError:
            raise ConfigError(
                f"{ENV_MAX_USERS}, {ENV_MAX_ACTIVE} and {ENV_MAX_JOBS} must be whole numbers"
            ) from None
        store = UserStore(url)
        root = Path(env.get(ENV_USERS_DIR, "") or Path.home() / "gpurelay-users")
        spaces = UserSpaces(store, Vault.from_env(env), root)
        mailer = Mailer.from_env(env)
        children = Children(spaces, max_active=max(1, max_active), on_jobs=Notices(store, mailer))
        return cls(store=store, signer=SessionSigner(key.encode()), spaces=spaces,
                   children=children, mailer=mailer, max_users=max(0, max_users),
                   max_jobs=max(1, max_jobs),
                   invite_code=env.get(ENV_INVITE_CODE, "").strip() or None)  # fmt: skip


def _client_id(request: Request) -> str:
    """The visitor's address; behind the host's proxy, the first ``X-Forwarded-For`` hop."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "?")


def _same_origin(request: Request) -> bool:
    """A form post must come from this site (no sign-in or sign-out from other sites)."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True  # old browsers and tools; the cookie is SameSite=Lax anyway
    return urlsplit(origin).netloc == request.headers.get("host", "")


def _page(name: str) -> str:
    return resources.files("gpurelay.gateway").joinpath(name).read_text("utf-8")


def _lifespan(children: Children | None) -> Any:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if children is None:
            yield
            return
        stop = threading.Event()

        def run() -> None:
            children.start_active()  # users whose jobs were in flight when the site stopped
            while not stop.wait(TICK_S):
                try:
                    children.tick()
                except Exception as exc:  # one bad pass must not end supervision
                    log.warning("checking users' GPU Relay failed: %s", exc)

        threading.Thread(target=run, name="gpurelay-children", daemon=True).start()
        try:
            yield
        finally:
            stop.set()
            children.stop_all()

    return lifespan


def create_gateway(settings: GatewaySettings) -> FastAPI:
    store, signer, children = settings.store, settings.signer, settings.children
    app = FastAPI(title="GPU Relay", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=_lifespan(children))  # fmt: skip

    def current_user(request: Request) -> User | None:
        verified = signer.verify(request.cookies.get(COOKIE_NAME))
        if verified is None:
            return None
        user = store.get(verified[0])
        if user is None or user.session_epoch != verified[1]:
            return None  # deleted, or signed out everywhere / password changed
        return user

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        if request.method == "POST" and not _same_origin(request):
            response: Any = JSONResponse({"detail": "cross-site request refused"}, 403)
        else:
            user = request.state.user = current_user(request)
            path = request.url.path
            is_api = path.startswith(("/api/", "/app/api/"))
            if user is None and path not in PUBLIC_PATHS:
                if is_api:
                    response = JSONResponse({"detail": "sign in first"}, status_code=401)
                else:
                    response = RedirectResponse("/signin", status_code=303)
            elif (user is not None and settings.mailer is not None and not user.email_verified
                  and path.startswith(NEEDS_VERIFIED)):  # fmt: skip
                if is_api or path.startswith("/api/setup"):
                    response = JSONResponse({"detail": "confirm your email first"}, 403)
                else:
                    response = RedirectResponse("/account?e=verify", status_code=303)
            else:
                response = await call_next(request)
        for name, value in HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    def set_session(response: Any, value: str, max_age: int) -> Any:
        response.set_cookie(
            COOKIE_NAME, value, max_age=max_age, httponly=True,
            secure=settings.secure_cookie, samesite="lax", path="/",
        )  # fmt: skip
        return response

    def with_session(response: Any, user: User) -> Any:
        return set_session(response, signer.issue(user.id, user.session_epoch),
                           int(signer.session_s))  # fmt: skip

    def signed_in(user: User) -> Any:
        return with_session(RedirectResponse("/", status_code=303), user)

    def back(path: str, error: str, email: str = "") -> Any:
        query = {"e": error, **({"email": email} if email else {})}
        return RedirectResponse(f"{path}?{urlencode(query)}", status_code=303)

    async def form(request: Request) -> dict[str, str]:
        raw = (await request.body())[:MAX_FORM].decode("utf-8", "replace")
        return {k: v[0] for k, v in parse_qs(raw).items()}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _page("home.html")

    @app.get("/signin", response_class=HTMLResponse)
    @app.get("/signup", response_class=HTMLResponse)
    def account_page() -> str:
        return _page("signin.html")

    @app.post("/signup")
    async def signup(request: Request) -> Any:
        client = _client_id(request)
        if settings.signup_limiter.blocked(client):
            return back("/signup", "busy")
        settings.signup_limiter.failed(client)  # every sign-up attempt counts
        data = await form(request)
        email = data.get("email", "")
        if settings.invite_code is not None and not hmac.compare_digest(
            data.get("invite", "").strip().encode(), settings.invite_code.encode()
        ):
            return back("/signup", "invite", email)
        if data.get("password", "") != data.get("confirm", ""):
            return back("/signup", "mismatch", email)
        if store.count() >= settings.max_users:
            return back("/signup", "full", email)
        try:
            user = await run_in_threadpool(store.create, email, data.get("password", ""))
        except AccountError as exc:
            return back("/signup", exc.code, email)
        if settings.mailer is not None:
            await run_in_threadpool(send_verification, store, settings.mailer, user)
        return signed_in(user)

    @app.post("/signin")
    async def signin(request: Request) -> Any:
        client = _client_id(request)
        if settings.signin_limiter.blocked(client):
            return back("/signin", "busy")
        data = await form(request)
        user = await run_in_threadpool(
            store.authenticate, data.get("email", ""), data.get("password", "")
        )
        if user is None:
            settings.signin_limiter.failed(client)
            return back("/signin", "wrong", data.get("email", ""))
        return signed_in(user)

    @app.post("/signout")
    def signout() -> Any:
        return set_session(RedirectResponse("/", status_code=303), "", 0)

    @app.post("/signout-everywhere")
    def signout_everywhere(request: Request) -> Any:
        store.end_sessions(request.state.user.id)
        return set_session(RedirectResponse("/", status_code=303), "", 0)

    @app.get("/api/me")
    def me(request: Request) -> dict[str, Any]:
        user: User | None = request.state.user
        return {
            "version": __version__,
            "user": None
            if user is None
            else {
                "email": user.email,
                "verified": user.email_verified or settings.mailer is None,
            },
            "mail": settings.mailer is not None,
            "signup": {
                "open": store.count() < settings.max_users,
                "invite": settings.invite_code is not None,
            },
        }

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(account_router(
        store, settings.mailer, _page, with_session,
        lambda response: set_session(response, "", 0), settings.spaces, children,
    ))  # fmt: skip

    if settings.spaces is not None:
        from gpurelay.gateway.setup_routes import setup_router

        @app.get("/setup", response_class=HTMLResponse)
        def setup_page() -> str:
            return _page("setup.html")

        def after_change(user_id: str) -> None:
            settings.after_change(user_id)
            if children is not None:
                children.restart(user_id)  # a running GPU Relay picks up the change

        app.include_router(setup_router(settings.spaces, after_change))

    if children is not None:
        from gpurelay.gateway.proxy import app_router

        app.include_router(app_router(children, settings.proxy_client, settings.max_jobs))

    return app
