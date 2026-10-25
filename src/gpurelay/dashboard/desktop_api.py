"""Desktop-only parts of the web app (plan D2): the setup wizard and controller control.

Served only by the desktop app on 127.0.0.1, behind a per-launch key: the window opens
``/?k=<key>``, which sets a cookie, and every other request needs that cookie. Other local
programs and websites cannot use these endpoints. ``gpurelay serve`` never includes them:
on a server, tokens belong in the host's secret settings.

Tokens typed into the wizard go straight to the OS keyring (Hugging Face) or the official
Kaggle token file, never into config.toml, logs or responses.
"""

from __future__ import annotations

import hmac
import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from gpurelay.config import Config, load_config, save_config
from gpurelay.constants import CONFIG_FILENAME
from gpurelay.errors import GpuRelayError

LAUNCH_COOKIE = "gpurelay_launch"
ACTION_HEADER = "X-GPURelay-Action"
ACCELERATORS = ("NvidiaTeslaT4", "NvidiaTeslaP100")


class DesktopHooks(Protocol):
    """What the desktop app does for the web app."""

    home: Path

    def controller_state(self) -> dict[str, Any]: ...
    def start_controller(self) -> None: ...
    def stop_controller(self) -> None: ...
    def reconfigured(self) -> None: ...
    def connect_server(self, url: str) -> None: ...
    def autostart(self) -> bool: ...
    def set_autostart(self, enabled: bool) -> None: ...
    def start_menu(self) -> bool: ...
    def set_start_menu(self, enabled: bool) -> None: ...
    def update_info(self) -> dict[str, Any]: ...
    def remove_from_windows(self) -> list[str]: ...
    def packaged(self) -> bool: ...


def _check_server(url: str) -> None:
    """The server answers ``/healthz`` like GPU Relay (a sleeping free server may take a
    minute to wake)."""
    try:
        with urllib.request.urlopen(f"{url}/healthz", timeout=90) as response:
            ok = json.loads(response.read(4096)).get("ok") is True
    except Exception as exc:
        raise GpuRelayError(f"could not reach {url}: {exc}") from exc
    if not ok:
        raise GpuRelayError(f"{url} does not look like a GPU Relay server")


def _kaggle_token_path() -> Path:
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    return (Path(config_dir) if config_dir else Path.home() / ".kaggle") / "access_token"


def _default_hf_api(token: str | None) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def _default_kaggle_client() -> Any:
    from gpurelay.backends.kaggle import OfficialKaggleClient

    return OfficialKaggleClient()


def _get_secret(name: str, env: str) -> str | None:
    from gpurelay.credentials import get_secret

    return get_secret(name, env)


def _set_secret(name: str, value: str) -> bool:
    from gpurelay.credentials import set_secret

    return set_secret(name, value)


def _kaggle_present() -> bool:
    from gpurelay.credentials import kaggle_credentials_present

    return kaggle_credentials_present()


def _modal_config_path() -> Path:
    return Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml")


def _modal_present() -> bool:
    from gpurelay.backends.modal import modal_credentials_present

    return modal_credentials_present()


@dataclass
class SetupDeps:
    """Outside services, replaceable in tests."""

    hf_api: Callable[[str | None], Any] = _default_hf_api
    kaggle_client: Callable[[], Any] = _default_kaggle_client
    check_server: Callable[[str], None] = _check_server
    kaggle_token_path: Callable[[], Path] = _kaggle_token_path
    kaggle_present: Callable[[], bool] = _kaggle_present
    get_secret: Callable[[str, str], str | None] = _get_secret
    set_secret: Callable[[str, str], bool] = _set_secret
    modal_config_path: Callable[[], Path] = _modal_config_path
    modal_present: Callable[[], bool] = _modal_present


class ServerForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(max_length=300)


class HfForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str | None = Field(default=None, max_length=300)
    repo: str | None = Field(default=None, max_length=100)


class KaggleForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kaggle_token: str | None = Field(default=None, max_length=300)
    session_token: str | None = Field(default=None, max_length=300)
    weekly_hours: float | None = Field(default=None, gt=0, le=168)
    accelerator: str | None = None


class ModalForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    terms_reviewed: bool = False
    token_id: str | None = Field(default=None, max_length=200)
    token_secret: str | None = Field(default=None, max_length=200)
    gpu: str = Field(default="T4", max_length=20)


class ColabForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


class FinishForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    autostart: bool = False


def _clean_token(value: str | None, what: str) -> str | None:
    if value is None or not value.strip():
        return None
    token = value.strip()
    if any(c.isspace() for c in token) or not token.isprintable():
        raise GpuRelayError(f"the {what} must be one line without spaces")
    return token


def add_launch_key(app: FastAPI, key: str) -> None:
    """Every request needs the per-launch cookie, except ``/healthz`` and the entry URL."""

    @app.middleware("http")
    async def launch_key(request: Request, call_next: Any) -> Any:
        given = request.query_params.get("k")
        if given is not None and hmac.compare_digest(given, key):
            response = RedirectResponse(request.url.path or "/", status_code=303)
            response.set_cookie(LAUNCH_COOKIE, key, httponly=True, samesite="strict", path="/")
            return response
        cookie = request.cookies.get(LAUNCH_COOKIE, "")
        if request.url.path != "/healthz" and not hmac.compare_digest(cookie, key):
            return JSONResponse({"detail": "open GPU Relay from its window"}, status_code=403)
        return await call_next(request)


def desktop_router(hooks: DesktopHooks, deps: SetupDeps | None = None) -> APIRouter:
    deps = deps or SetupDeps()
    router = APIRouter()
    home = hooks.home
    config_path = home / CONFIG_FILENAME

    def act(value: str | None) -> None:
        if value != "1":
            raise HTTPException(status_code=403, detail=f"missing {ACTION_HEADER} header")

    get_secret, set_secret, kaggle_present = deps.get_secret, deps.set_secret, deps.kaggle_present

    def config() -> Config:
        return load_config(config_path)

    def guarded(fn: Callable[[], dict[str, Any]]) -> Any:
        try:
            return fn()
        except GpuRelayError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/desktop/status")
    def status() -> dict[str, Any]:
        from gpurelay.credentials import HF_TOKEN
        from gpurelay.desktop.settings import DesktopSettings

        settings = DesktopSettings.load(home)
        cfg = config()
        return {
            "mode": settings.mode,
            "server_url": settings.server_url,
            "configured": config_path.exists(),
            "storage": {
                "kind": cfg.storage.kind,
                "repo": cfg.storage.hf_repo_id,
                "token": get_secret(HF_TOKEN, "HF_TOKEN") is not None,
            },
            "kaggle": {
                "enabled": cfg.backends.kaggle.enabled,
                "username": cfg.backends.kaggle.username,
                "weekly_hours": cfg.backends.kaggle.weekly_gpu_hours,
                "accelerator": cfg.backends.kaggle.accelerator,
                "credentials": kaggle_present(),
            },
            "colab": {"enabled": cfg.backends.colab_manual.enabled},
            "modal": {
                "enabled": cfg.backends.modal.enabled,
                "gpu": cfg.backends.modal.gpu,
                "credentials": deps.modal_present(),
                "monthly_usd": cfg.backends.modal.monthly_credit_usd,
            },
            "local": {"enabled": cfg.backends.local.enabled},
            "controller": hooks.controller_state(),
            "autostart": hooks.autostart(),
            "packaged": hooks.packaged(),
            "start_menu": hooks.start_menu(),
            "update": hooks.update_info(),
        }

    @router.post("/api/setup/server")
    async def setup_server(
        form: ServerForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        from gpurelay.desktop.settings import normalize_server_url

        def run() -> dict[str, Any]:
            url = normalize_server_url(form.url)
            deps.check_server(url)
            hooks.connect_server(url)
            return {"ok": True, "url": url}

        return await run_in_threadpool(guarded, run)

    @router.post("/api/setup/local-storage")
    def setup_local(
        x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER),
    ) -> Any:
        act(x_gpurelay_action)

        def run() -> dict[str, Any]:
            cfg = config()
            cfg.storage.kind = "localfs"
            cfg.backends.local.enabled = True
            save_config(cfg, config_path)
            return {"ok": True, "store": str(cfg.storage.localfs_root or home / "store")}

        return guarded(run)

    @router.post("/api/setup/hf")
    async def setup_hf(
        form: HfForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        from gpurelay.controller.setup import setup_storage
        from gpurelay.credentials import HF_TOKEN

        def run() -> dict[str, Any]:
            token = _clean_token(form.token, "Hugging Face token") or get_secret(
                HF_TOKEN, "HF_TOKEN"
            )
            if not token:
                raise GpuRelayError("paste a Hugging Face token with write access")
            if not token.startswith("hf_"):
                raise GpuRelayError("a Hugging Face token starts with hf_")
            api = deps.hf_api(token)
            try:
                api.whoami()
            except Exception as exc:
                raise GpuRelayError(f"Hugging Face did not accept this token: {exc}") from exc
            cfg = config()
            try:
                result = setup_storage(cfg, api, repo=(form.repo or "").strip() or None)
            except GpuRelayError:
                raise
            except Exception as exc:
                raise GpuRelayError(f"could not set up the storage repo: {exc}") from exc
            stored = set_secret(HF_TOKEN, token)
            save_config(cfg, config_path)
            return {"ok": True, "repo": result.repo_id, "user": result.hf_user,
                    "created": result.created, "keyring": stored}  # fmt: skip

        return await run_in_threadpool(guarded, run)

    @router.post("/api/setup/kaggle")
    async def setup_kaggle_route(
        form: KaggleForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        from gpurelay.controller.setup import setup_kaggle, validate_session_token
        from gpurelay.credentials import HF_TOKEN

        def run() -> dict[str, Any]:
            cfg = config()
            if not cfg.storage.hf_repo_id:
                raise GpuRelayError("set up Hugging Face storage first")
            kaggle_token = _clean_token(form.kaggle_token, "Kaggle token")
            if kaggle_token:
                path = deps.kaggle_token_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(kaggle_token, encoding="utf-8")
                if os.name == "posix":
                    path.chmod(0o600)
            if not kaggle_present():
                raise GpuRelayError(
                    "paste your Kaggle API token (kaggle.com > Settings > API > Create New Token)"
                )
            session = _clean_token(form.session_token, "Hugging Face token") or get_secret(
                HF_TOKEN, "HF_TOKEN"
            )
            if not session:
                raise GpuRelayError("a Hugging Face token for Kaggle sessions is needed")
            validate_session_token(session, cfg.storage.hf_repo_id, deps.hf_api(None))
            try:
                result = setup_kaggle(cfg, deps.kaggle_client(), session_token=session,
                                      repo_id=cfg.storage.hf_repo_id)  # fmt: skip
            except GpuRelayError:
                raise
            except Exception as exc:
                raise GpuRelayError(f"Kaggle setup failed: {exc}") from exc
            if form.weekly_hours is not None:
                cfg.backends.kaggle.weekly_gpu_hours = form.weekly_hours
            if form.accelerator is not None:
                if form.accelerator not in ACCELERATORS:
                    raise GpuRelayError(f"GPU must be one of {', '.join(ACCELERATORS)}")
                cfg.backends.kaggle.accelerator = form.accelerator
            save_config(cfg, config_path)
            return {"ok": True, "username": result.username, "created": result.created}

        return await run_in_threadpool(guarded, run)

    @router.post("/api/setup/modal")
    def setup_modal(
        form: ModalForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)

        def run() -> dict[str, Any]:
            cfg = config()
            if form.enabled and not cfg.storage.hf_repo_id:
                raise GpuRelayError("Modal sessions need Hugging Face storage (step 2)")
            if form.enabled and not form.terms_reviewed:
                raise GpuRelayError("confirm that you read Modal's terms of service")
            token_id = _clean_token(form.token_id, "Modal token id")
            token_secret = _clean_token(form.token_secret, "Modal token secret")
            if token_id or token_secret:
                if not (token_id and token_secret):
                    raise GpuRelayError("paste both the Modal token id and the token secret")
                path = deps.modal_config_path()
                if path.exists():
                    raise GpuRelayError(
                        f"{path} already exists: run `modal token set` to change the token"
                    )
                path.write_text(  # Modal's own config file format
                    f'[default]\ntoken_id = "{token_id}"\ntoken_secret = "{token_secret}"\n'
                    "active = true\n",
                    encoding="utf-8",
                )
                if os.name == "posix":
                    path.chmod(0o600)
            if form.enabled and not deps.modal_present():
                raise GpuRelayError(
                    "paste a Modal token (modal.com > Settings > API Tokens > New Token)"
                )
            from gpurelay.config import ModalSettings

            cfg.backends.modal = ModalSettings(
                **{
                    **cfg.backends.modal.model_dump(),
                    "enabled": form.enabled,
                    "terms_reviewed": form.terms_reviewed or cfg.backends.modal.terms_reviewed,
                    "gpu": form.gpu.strip() or "T4",
                }
            )
            save_config(cfg, config_path)
            return {"ok": True, "enabled": form.enabled}

        return guarded(run)

    @router.post("/api/setup/colab")
    def setup_colab(
        form: ColabForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        cfg = config()
        cfg.backends.colab_manual.enabled = form.enabled
        save_config(cfg, config_path)
        return {"ok": True}

    @router.post("/api/setup/finish")
    def setup_finish(
        form: FinishForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        from gpurelay.desktop.settings import LOCAL, DesktopSettings

        if not config_path.exists():
            raise HTTPException(status_code=400, detail="choose where checkpoints live first")
        settings = DesktopSettings.load(home)
        settings.mode = LOCAL
        settings.save(home)
        try:
            hooks.set_autostart(form.autostart)
        except Exception as exc:  # autostart is a convenience: setup still finishes
            return {"ok": True, "warning": f"could not set autostart: {exc}"}
        finally:
            hooks.reconfigured()
        return {"ok": True}

    @router.post("/api/desktop/autostart")
    def autostart(
        form: FinishForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        hooks.set_autostart(form.autostart)
        return {"ok": True, "autostart": hooks.autostart()}

    @router.post("/api/desktop/start-menu")
    def start_menu(
        form: ColabForm, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        if not hooks.packaged():
            raise HTTPException(
                status_code=400, detail="only the packaged app has a Start menu entry"
            )
        try:
            hooks.set_start_menu(form.enabled)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not change it: {exc}") from exc
        return {"ok": True, "start_menu": hooks.start_menu()}

    @router.post("/api/desktop/remove")
    def remove(x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)) -> Any:
        act(x_gpurelay_action)
        return {"ok": True, "removed": hooks.remove_from_windows()}

    @router.post("/api/controller/{action}")
    def controller(
        action: str, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> Any:
        act(x_gpurelay_action)
        if action == "start":
            hooks.start_controller()
        elif action == "stop":
            hooks.stop_controller()
        else:
            raise HTTPException(status_code=404, detail=f"unknown action {action!r}")
        return {"ok": True, "controller": hooks.controller_state()}

    return router


def create_setup_app(
    hooks: DesktopHooks, *, launch_key: str, error: str | None = None,
    deps: SetupDeps | None = None,
) -> FastAPI:  # fmt: skip
    """The web app before GPU Relay is set up (or when its config cannot be used): the
    page shows only the setup wizard."""
    app = FastAPI(title="GPU Relay", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("gpurelay.dashboard").joinpath("index.html").read_text("utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/me")
    def me() -> dict[str, Any]:
        return {"role": "owner", "login": False, "anonymous": False, "desktop": True,
                "configured": False, "error": error}  # fmt: skip

    app.include_router(desktop_router(hooks, deps))
    add_launch_key(app, launch_key)
    return app
