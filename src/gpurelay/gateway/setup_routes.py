"""The website's setup steps for a signed-in user (plan M2).

``GET /api/setup`` says what is connected (never a token). Each ``POST`` checks a token with
the platform before keeping it. ``after_change(user_id)`` runs after any change, so the
user's running GPU Relay picks it up (plan M3).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gpurelay.dashboard.auth import RateLimiter
from gpurelay.dashboard.desktop_api import ACCELERATORS
from gpurelay.errors import GpuRelayError
from gpurelay.gateway.space import UserSpaces
from gpurelay.gateway.userenv import HF, KAGGLE, MODAL_ID, MODAL_SECRET

TOKEN_MAX = 4096  # a kaggle.json pasted whole is longer than an API token


class HfForm(BaseModel):
    token: str = Field(min_length=1, max_length=400)
    repo: str | None = Field(default=None, max_length=200)


class KaggleForm(BaseModel):
    token: str | None = Field(default=None, max_length=TOKEN_MAX)
    accelerator: str | None = Field(default=None, max_length=40)


class ModalForm(BaseModel):
    token_id: str = Field(min_length=1, max_length=400)
    token_secret: str = Field(min_length=1, max_length=400)
    terms: bool = False
    gpu: str | None = Field(default=None, max_length=40)


class ColabForm(BaseModel):
    enabled: bool
    first: bool = False
    wait_min: int = Field(default=30, ge=5, le=24 * 60)


class LimitsForm(BaseModel):
    modal_paid_cap_usd: float = Field(default=0.0, ge=0, le=10_000)
    kaggle_weekly_hours: float | None = Field(default=None, ge=0, le=168)


class RemoveForm(BaseModel):
    name: str = Field(pattern="^(hf|kaggle|modal)$")


def _one_line(value: str, what: str) -> str:
    value = value.strip()
    if not value or any(c.isspace() for c in value) or not value.isprintable():
        raise GpuRelayError(f"the {what} must be one line without spaces")
    return value


def _kaggle_token(value: str) -> str:
    """An API token (KGAT_...), or a kaggle.json pasted whole (username and key)."""
    value = value.strip()
    if value.startswith("{"):
        try:
            data = json.loads(value)
            return json.dumps({"username": str(data["username"]), "key": str(data["key"])})
        except (ValueError, KeyError, TypeError):
            raise GpuRelayError("that kaggle.json has no username and key") from None
    return _one_line(value, "Kaggle token")


def setup_router(
    spaces: UserSpaces, after_change: Callable[[str], None] = lambda user_id: None
) -> APIRouter:
    router = APIRouter()
    limiter = RateLimiter(per_client=20, total=500, window_s=3600.0)  # checks start processes

    def user_id(request: Request) -> str:
        return str(request.state.user.id)

    async def guarded(request: Request, work: Callable[[str], dict[str, Any]]) -> Any:
        uid = user_id(request)
        if limiter.blocked(uid):
            return JSONResponse({"detail": "too many setup attempts; try again in an hour"}, 429)
        limiter.failed(uid)  # every check counts
        try:
            result = await run_in_threadpool(work, uid)
        except GpuRelayError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        await run_in_threadpool(after_change, uid)  # may restart the user's GPU Relay
        return {"ok": True, **result}

    @router.get("/api/setup")
    def status(request: Request) -> dict[str, Any]:
        uid = user_id(request)
        config, saved = spaces.config(uid), spaces.saved(uid)
        repo = config.storage.hf_repo_id if config.storage.kind == "hf_hub" else None
        kaggle, modal, colab = (
            config.backends.kaggle,
            config.backends.modal,
            config.backends.colab_manual,
        )
        hf_ok = HF in saved and bool(repo)
        steps: dict[str, dict[str, Any]] = {
            "hf": {"connected": hf_ok, "repo": repo, "saved_at": saved.get(HF)},
            "kaggle": {"connected": KAGGLE in saved and kaggle.enabled,
                       "username": kaggle.username, "accelerator": kaggle.accelerator,
                       "weekly_hours": kaggle.weekly_gpu_hours, "saved_at": saved.get(KAGGLE)},
            "modal": {"connected": MODAL_SECRET in saved and modal.enabled, "gpu": modal.gpu,
                      "paid_cap_usd": modal.monthly_paid_cap_usd,
                      "saved_at": saved.get(MODAL_SECRET)},
            "colab": {"enabled": colab.enabled, "first": colab.first,
                      "wait_min": colab.wait_s // 60},
        }  # fmt: skip
        unreadable = spaces.unreadable(uid)
        return {"steps": steps, "ready": spaces.ready(uid), "accelerators": ACCELERATORS,
                "unreadable": unreadable}  # fmt: skip

    @router.post("/api/setup/hf")
    async def setup_hf(form: HfForm, request: Request) -> Any:
        def work(uid: str) -> dict[str, Any]:
            token = _one_line(form.token, "Hugging Face token")
            if not token.startswith("hf_"):
                raise GpuRelayError("a Hugging Face token starts with hf_")
            result = spaces.run_step(uid, "hf", {"repo": form.repo}, {HF: token})
            if KAGGLE in spaces.saved(uid):  # Kaggle sessions read the token: refresh it there
                try:
                    spaces.run_step(uid, "kaggle", {"rotate": True}, {})
                except GpuRelayError as exc:
                    result["kaggle_warning"] = f"update Kaggle again: {exc}"
            return result

        return await guarded(request, work)

    @router.post("/api/setup/kaggle")
    async def setup_kaggle(form: KaggleForm, request: Request) -> Any:
        def work(uid: str) -> dict[str, Any]:
            if form.accelerator is not None and form.accelerator not in ACCELERATORS:
                raise GpuRelayError(f"GPU must be one of {', '.join(ACCELERATORS)}")
            new = {KAGGLE: _kaggle_token(form.token)} if form.token else {}
            if not new and KAGGLE not in spaces.saved(uid):
                raise GpuRelayError("paste your Kaggle API token (kaggle.com > Settings > API)")
            # rotate: the secrets dataset gets the Hugging Face token this site holds now
            request_ = {"accelerator": form.accelerator, "rotate": True}
            return spaces.run_step(uid, "kaggle", request_, new)

        return await guarded(request, work)

    @router.post("/api/setup/modal")
    async def setup_modal(form: ModalForm, request: Request) -> Any:
        def work(uid: str) -> dict[str, Any]:
            if not form.terms:
                raise GpuRelayError("confirm that you read Modal's terms of service")
            new = {MODAL_ID: _one_line(form.token_id, "Modal token id"),
                   MODAL_SECRET: _one_line(form.token_secret, "Modal token secret")}  # fmt: skip
            return spaces.run_step(uid, "modal", {"terms": True, "gpu": form.gpu}, new)

        return await guarded(request, work)

    @router.post("/api/setup/colab")
    async def setup_colab(form: ColabForm, request: Request) -> Any:
        def work(uid: str) -> dict[str, Any]:
            config = spaces.config(uid)
            if form.enabled and not config.storage.hf_repo_id:
                raise GpuRelayError("connect Hugging Face first")
            colab = config.backends.colab_manual
            colab.enabled, colab.first = form.enabled, form.enabled and form.first
            colab.wait_s = form.wait_min * 60
            spaces.save_config(uid, config)
            return {}

        return await guarded(request, work)

    @router.post("/api/setup/limits")
    async def setup_limits(form: LimitsForm, request: Request) -> Any:
        def work(uid: str) -> dict[str, Any]:
            config = spaces.config(uid)
            config.backends.modal.monthly_paid_cap_usd = form.modal_paid_cap_usd
            if form.kaggle_weekly_hours is not None:
                config.backends.kaggle.weekly_gpu_hours = form.kaggle_weekly_hours
            spaces.save_config(uid, config)
            return {}

        return await guarded(request, work)

    @router.post("/api/setup/remove")
    async def remove(form: RemoveForm, request: Request) -> Any:
        def work(uid: str) -> dict[str, Any]:
            config = spaces.config(uid)
            if form.name in ("kaggle", "hf"):
                config.backends.kaggle.enabled = False
                spaces.remove_secrets(uid, (KAGGLE,))
            if form.name in ("modal", "hf"):
                config.backends.modal.enabled = False
                spaces.remove_secrets(uid, (MODAL_ID, MODAL_SECRET))
            if form.name == "hf":  # everything needs storage
                config.backends.colab_manual.enabled = False
                spaces.remove_secrets(uid, (HF,))
            spaces.save_config(uid, config)
            return {}

        return await guarded(request, work)

    return router
