"""Account care (plan M4): email verification, password reset and change, notifications,
and deleting the account.

One-time links carry a random token; the database keeps only its SHA-256, so a copy of the
database cannot be used to take over an account. Verification links last a day, reset links
an hour, and each works once. "Forgot password" always answers the same way, so it does not
reveal which emails have accounts. Deleting an account removes its tokens, settings and
files on this server; the user's own storage repo and Kaggle dataset stay theirs.
"""

from __future__ import annotations

import hashlib
import secrets
import shutil
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from gpurelay.accounts import passwords
from gpurelay.accounts.store import AccountError, User, UserStore
from gpurelay.dashboard.auth import RateLimiter
from gpurelay.gateway.children import Children
from gpurelay.gateway.mail import Mailer
from gpurelay.gateway.space import UserSpaces

VERIFY, RESET = "verify", "reset"
VERIFY_S, RESET_S = 24 * 3600.0, 3600.0


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_link(store: UserStore, user_id: str, kind: str, ttl_s: float) -> str:
    token = secrets.token_urlsafe(32)
    store.put_link(token_hash(token), user_id, kind, store.clock() + ttl_s)
    return token


def send_verification(store: UserStore, mailer: Mailer, user: User) -> bool:
    token = new_link(store, user.id, VERIFY, VERIFY_S)
    return mailer.send(user.email, "Confirm your email for GPU Relay", (
        "Hello,\n\nconfirm that this is your email address for GPU Relay:\n\n"
        f"{mailer.link('/verify?t=' + token)}\n\n"
        "The link works once, for 24 hours. If you did not create an account, ignore this mail.\n"
    ))  # fmt: skip


class PasswordForm(BaseModel):
    current: str = Field(max_length=passwords.MAX_LEN)
    new: str = Field(max_length=passwords.MAX_LEN)


class UsernameForm(BaseModel):
    username: str = Field(max_length=64)


class NotifyForm(BaseModel):
    notify: bool


class DeleteForm(BaseModel):
    password: str = Field(max_length=passwords.MAX_LEN)


def account_router(
    store: UserStore,
    mailer: Mailer | None,
    page: Callable[[str], str],
    issue_session: Callable[[Any, User], Any],
    end_session: Callable[[Any], Any],
    spaces: UserSpaces | None = None,
    children: Children | None = None,
) -> APIRouter:
    router = APIRouter()
    mail_limiter = RateLimiter(per_client=5, total=200, window_s=3600.0)
    check_limiter = RateLimiter(per_client=5, total=100, window_s=900.0)

    def me(request: Request) -> User:
        user: User = request.state.user
        return user

    async def form(request: Request) -> dict[str, str]:
        raw = (await request.body())[:4096].decode("utf-8", "replace")
        return {k: v[0] for k, v in parse_qs(raw).items()}

    def client(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        return forwarded or (request.client.host if request.client else "?")

    @router.get("/account", response_class=HTMLResponse)
    def account_page() -> str:
        return page("account.html")

    @router.get("/api/account")
    def account(request: Request) -> dict[str, Any]:
        user = me(request)
        return {"email": user.email, "username": user.username,
                "verified": user.email_verified, "notify": user.notify,
                "mail": mailer is not None}  # fmt: skip

    @router.post("/api/account/verify-email")
    async def resend(request: Request) -> Any:
        user = me(request)
        if mailer is None:
            return JSONResponse({"detail": "this site sends no email"}, status_code=400)
        if user.email_verified:
            return {"ok": True}
        if mail_limiter.blocked(user.id):
            return JSONResponse({"detail": "too many mails; try again in an hour"}, 429)
        mail_limiter.failed(user.id)
        sent = await run_in_threadpool(send_verification, store, mailer, user)
        if not sent:
            return JSONResponse({"detail": "the mail could not be sent; try later"}, 502)
        return {"ok": True}

    @router.get("/verify")
    def verify(t: str = "") -> Any:
        user_id = store.take_link(token_hash(t), VERIFY) if t else None
        if user_id is None:
            return RedirectResponse("/account?e=link", status_code=303)
        store.set_email_verified(user_id)
        return RedirectResponse("/account?m=verified", status_code=303)

    @router.get("/forgot", response_class=HTMLResponse)
    @router.get("/reset", response_class=HTMLResponse)
    def reset_page() -> str:
        return page("reset.html")

    @router.post("/forgot")
    async def forgot(request: Request) -> Any:
        if mailer is None:
            return RedirectResponse("/forgot?e=nomail", status_code=303)
        who = client(request)
        if mail_limiter.blocked(who):
            return RedirectResponse("/forgot?e=busy", status_code=303)
        mail_limiter.failed(who)
        user = store.find((await form(request)).get("email", ""))
        if user is not None:
            token = new_link(store, user.id, RESET, RESET_S)
            await run_in_threadpool(mailer.send, user.email, "Reset your GPU Relay password", (
                "Hello,\n\nsomeone asked to reset the password of your GPU Relay account. "
                f"Choose a new one here:\n\n{mailer.link('/reset?t=' + token)}\n\n"
                "The link works once, for one hour. If it was not you, ignore this mail: your "
                "password stays the same.\n"
            ))  # fmt: skip
        return RedirectResponse("/forgot?m=sent", status_code=303)  # the same answer for all

    @router.post("/reset")
    async def reset(request: Request) -> Any:
        data = await form(request)
        token, new, confirm = data.get("t", ""), data.get("password", ""), data.get("confirm", "")
        if new != confirm:
            return RedirectResponse(f"/reset?t={token}&e=mismatch", status_code=303)
        if passwords.problem(new):
            return RedirectResponse(f"/reset?t={token}&e=password", status_code=303)
        user_id = store.take_link(token_hash(token), RESET) if token else None
        if user_id is None:
            return RedirectResponse("/forgot?e=link", status_code=303)
        store.set_password(user_id, new)  # also ends every session
        store.set_email_verified(user_id)  # the link proved the address
        return RedirectResponse("/signin?m=reset", status_code=303)

    @router.post("/api/account/password")
    async def change_password(form_: PasswordForm, request: Request) -> Any:
        user = me(request)
        if check_limiter.blocked(user.id):
            return JSONResponse({"detail": "too many attempts; try again later"}, 429)
        if await run_in_threadpool(store.authenticate, user.email, form_.current) is None:
            check_limiter.failed(user.id)
            return JSONResponse({"detail": "the current password is wrong"}, status_code=400)
        try:
            await run_in_threadpool(store.set_password, user.id, form_.new)
        except AccountError:
            return JSONResponse({"detail": "use a new password of 12 to 256 characters"}, 400)
        fresh = store.get(user.id)
        assert fresh is not None
        return issue_session(JSONResponse({"ok": True}), fresh)  # other devices signed out

    @router.post("/api/account/username")
    def set_username(form_: UsernameForm, request: Request) -> Any:
        try:
            name = store.set_username(me(request).id, form_.username)
        except AccountError as exc:
            return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=400)
        return {"ok": True, "username": name}

    @router.post("/api/account/notify")
    def set_notify(form_: NotifyForm, request: Request) -> dict[str, bool]:
        store.set_notify(me(request).id, form_.notify)
        return {"ok": True}

    @router.post("/api/account/delete")
    async def delete(form_: DeleteForm, request: Request) -> Any:
        user = me(request)
        if check_limiter.blocked(user.id):
            return JSONResponse({"detail": "too many attempts; try again later"}, 429)
        if await run_in_threadpool(store.authenticate, user.email, form_.password) is None:
            check_limiter.failed(user.id)
            return JSONResponse({"detail": "the password is wrong"}, status_code=400)

        def remove() -> None:
            if children is not None:
                children.stop(user.id)  # running GPU sessions finish on their platform
            if spaces is not None:
                shutil.rmtree(spaces.root / user.id, ignore_errors=True)
            store.delete(user.id)

        await run_in_threadpool(remove)
        return end_session(JSONResponse({"ok": True}))

    return router
