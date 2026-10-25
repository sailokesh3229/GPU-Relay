"""``gpurelay dashboard``: local web UI (spec §15, plan Phase 8).

FastAPI plus one static page, no build step. It never runs the controller: it reads the state
database and storage, and records pause/resume/cancel requests that the running controller
applies (exactly like the CLI). Served on localhost by ``gpurelay dashboard``; on the internet
only behind a login (``auth``, used by ``gpurelay serve``).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from gpurelay import __version__
from gpurelay.controller.estimator import fit_message
from gpurelay.controller.lock import controller_running
from gpurelay.controller.service import (
    Context,
    colab_session,
    correct_quota,
    merged_log,
    request,
    resolve_job,
    submit_job,
)
from gpurelay.controller.views import JobRow, backend_rows, budget_lines, job_rows
from gpurelay.dashboard.auth import COOKIE_NAME, OWNER, VIEWER, Auth
from gpurelay.dashboard.forms import (
    FormError,
    QuotaForm,
    SubmitForm,
    UploadForm,
    daily_usage,
    parse_body,
    save_upload,
)
from gpurelay.durations import parse_duration
from gpurelay.errors import GpuRelayError
from gpurelay.models import DataSpec, Job, JobState, ResumabilityLevel, Segment
from gpurelay.storage.layout import final_prefix

ACTION_HEADER = "X-GPURelay-Action"  # a custom header: other sites cannot send it (no CORS)
ACTIONS = ("pause", "resume", "cancel")
LOG_TAIL_DEFAULT = 20_000
LOG_TAIL_MAX = 500_000


def _iso(value: datetime | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float | int):
        value = datetime.fromtimestamp(value, timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _job_json(row: JobRow) -> dict[str, Any]:
    job = row.job
    return {
        "id": job.id,
        "name": job.name,
        "kind": job.kind.value,
        "state": job.state.value,
        "terminal": job.state.is_terminal,
        "message": job.message,
        "backend": None if row.backend == "-" else row.backend,
        "progress": None if row.progress == "-" else row.progress,
        "completed": row.completed,
        "total": row.total,
        "used_h": row.used_h,
        "eta_h": row.eta_h,
        "eta_basis": row.eta_basis,
        "segments": row.segments,
        "priority": job.priority,
        "submitted_at": _iso(job.submitted_at),
        "requested": job.requested,
        "entry": job.entry,
        "prefer_backend": job.prefer_backend,
        "estimate_h": job.estimate_h,
    }


def _segment_json(seg: Segment, now: float) -> dict[str, Any]:
    end = seg.ended_at.timestamp() if seg.ended_at else now
    return {
        "id": seg.id,
        "index": seg.index,
        "role": seg.role,
        "backend": seg.backend,
        "gpu": seg.gpu_model,
        "state": seg.state.value,
        "outcome": seg.outcome,
        "started_at": _iso(seg.started_at),
        "ended_at": _iso(seg.ended_at),
        "deadline_at": _iso(seg.deadline_at),
        "duration_s": end - seg.started_at.timestamp() if seg.started_at else 0.0,
        "start_checkpoint": seg.start_checkpoint_seq,
        "end_checkpoint": seg.end_checkpoint_seq,
        "message": seg.message,
    }


LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
PUBLIC_PATHS = frozenset({"/login", "/healthz"})
# Served behind a login on the internet: no framing by other sites (clickjacking), no caching.
HOSTED_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _client_id(request: Request) -> str:
    """The visitor's address; behind the host's proxy, the first ``X-Forwarded-For`` hop."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "?")


def _host_name(header: str) -> str:
    """``Host`` header without the port (IPv6 literals keep their brackets)."""
    if header.startswith("["):
        return header.split("]", 1)[0] + "]"
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def create_app(
    ctx: Context,
    *,
    clock: Callable[[], float] = time.time,
    allowed_hosts: frozenset[str] | None = LOCAL_HOSTS,
    auth: Auth | None = None,
) -> FastAPI:
    """``allowed_hosts``: accepted ``Host`` names. Anything else is refused, which stops DNS
    rebinding (a website pointing its own domain at 127.0.0.1 to reach this API). None accepts
    any host, for a server behind ``auth``.

    ``auth``: require a login; viewers may only read. Without it everyone is the owner."""
    if allowed_hosts is None and auth is None:
        raise ValueError("accepting any Host header needs a login (auth)")
    app = FastAPI(title="GPU Relay", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def check_login(request: Request, call_next: Any) -> Any:
        if auth is None:
            request.state.role = OWNER
            return await call_next(request)
        path = request.url.path
        role = auth.verify(request.cookies.get(COOKIE_NAME))
        request.state.role = role
        if role is None and path not in PUBLIC_PATHS:
            if path.startswith("/api/"):
                response: Any = JSONResponse({"detail": "log in first"}, status_code=401)
            else:
                response = RedirectResponse("/login", status_code=303)
        elif (
            role == VIEWER
            and request.method not in ("GET", "HEAD")
            and (path not in PUBLIC_PATHS and path != "/logout")
        ):
            response = JSONResponse({"detail": "view-only access"}, status_code=403)
        else:
            response = await call_next(request)
        response.headers.update(HOSTED_HEADERS)
        return response

    # Registered last so it runs first: a wrong Host is refused before anything else.
    @app.middleware("http")
    async def check_host(request: Request, call_next: Any) -> Any:
        if allowed_hosts is not None and (
            _host_name(request.headers.get("host", "")) not in allowed_hosts
        ):
            return JSONResponse({"detail": "unexpected Host header"}, status_code=421)
        return await call_next(request)

    def _set_session(response: Any, value: str, max_age: int) -> None:
        assert auth is not None
        response.set_cookie(
            COOKIE_NAME, value, max_age=max_age, httponly=True, secure=auth.secure_cookie,
            samesite="lax", path="/",
        )  # fmt: skip

    if auth is not None:

        @app.get("/login", response_class=HTMLResponse)
        def login_page() -> str:
            return resources.files("gpurelay.dashboard").joinpath("login.html").read_text("utf-8")

        @app.post("/login")
        async def login(request: Request) -> Any:
            client = _client_id(request)
            if auth.limiter.blocked(client):
                return RedirectResponse("/login?e=busy", status_code=303)
            form = parse_qs((await request.body())[:4096].decode("utf-8", "replace"))
            role = auth.check_password(form.get("password", [""])[0])
            if role is None:
                auth.limiter.failed(client)
                return RedirectResponse("/login?e=wrong", status_code=303)
            response = RedirectResponse("/", status_code=303)
            _set_session(response, auth.issue(role), int(auth.session_s))
            return response

        @app.post("/logout")
        def logout() -> Any:
            response = RedirectResponse("/login", status_code=303)
            _set_session(response, "", 0)
            return response

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/me")
    def me(request: Request) -> dict[str, Any]:
        return {"role": request.state.role, "login": auth is not None}

    controller = ctx.controller

    def find(job_id: str) -> Job:
        try:
            return resolve_job(ctx, job_id)
        except GpuRelayError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def row_for(job: Job, now: float) -> JobRow:
        return next(r for r in job_rows(controller, now) if r.job.id == job.id)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("gpurelay.dashboard").joinpath("index.html").read_text("utf-8")

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        now = clock()
        rows = sorted(job_rows(controller, now), key=lambda r: r.job.submitted_at, reverse=True)
        return {
            "now": _iso(now),
            "controller_running": controller_running(ctx.home),
            "jobs": [_job_json(r) for r in rows],
            "backends": [
                {
                    "name": b.name,
                    "automated": b.automated,
                    "gpu": b.gpu,
                    "budget": b.budget,
                    "budget_h": b.budget_h,
                    "used_h": b.used_h,
                    "remaining": b.remaining,
                    "remaining_h": b.remaining_h,
                    "estimated": b.estimated,
                    "next_reset": _iso(b.next_reset),
                    "running": b.running,
                    "max_concurrent": b.max_concurrent,
                    "daily_h": daily_usage(ctx.db.ledger(b.name), now),
                }
                for b in backend_rows(controller, now)
            ],
            "colab_actions": [
                {"job_id": r.job.id, "name": r.job.name, "command": f"gpurelay colab {r.job.id}"}
                for r in rows
                if r.job.state is JobState.AWAITING_USER
            ],
        }

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, Any]:
        now = clock()
        job = find(job_id)
        row = row_for(job, now)
        fit = None
        if not job.state.is_terminal:
            fit = fit_message(row.eta_h, budget_lines(controller, now))
        last = ctx.db.last_checkpoint(job.id)
        return {
            "job": _job_json(row),
            "checkpoints": ctx.db.checkpoint_count(job.id),
            "last_checkpoint": None if last is None else {"seq": last[0], "at": last[1]},
            "fit": fit,
            "segments": [_segment_json(s, now) for s in ctx.db.segments(job.id)],
            "events": [
                {"at": _iso(e.ts), "kind": e.kind, "message": e.message}
                for e in ctx.db.events(job.id, limit=30)
            ],
        }

    @app.get("/api/jobs/{job_id}/log")
    def job_log(
        job_id: str, tail: int = Query(LOG_TAIL_DEFAULT, ge=1, le=LOG_TAIL_MAX)
    ) -> dict[str, Any]:
        job = find(job_id)
        try:
            text = merged_log(ctx, job)
        except Exception as exc:  # storage unreachable: report, do not crash the page
            raise HTTPException(status_code=502, detail=f"log unavailable: {exc}") from exc
        return {"text": text[-tail:], "length": len(text), "final": job.state is JobState.COMPLETED}

    @app.get("/api/jobs/{job_id}/colab-notebook")
    def colab_notebook(job_id: str) -> FileResponse:
        """The job's Colab resume notebook (useful when the controller runs on a VM)."""
        find(job_id)
        try:
            info = colab_session(ctx, job_id)
        except GpuRelayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return FileResponse(info.notebook, media_type="application/x-ipynb+json",
                            filename=info.notebook.name)  # fmt: skip

    def require_action_header(value: str | None) -> None:
        if value != "1":  # a custom header: a form on another site cannot send it
            raise HTTPException(status_code=403, detail=f"missing {ACTION_HEADER} header")

    @app.get("/api/jobs/{job_id}/files")
    def job_files(job_id: str) -> dict[str, Any]:
        job = find(job_id)
        if job.state is not JobState.COMPLETED:
            return {"completed": False, "files": []}
        prefix = final_prefix(job.id) + "/"
        try:
            keys = ctx.storage.list(prefix)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"storage unavailable: {exc}") from exc
        return {"completed": True, "files": sorted(k[len(prefix) :] for k in keys)}

    @app.get("/api/jobs/{job_id}/files/{name:path}")
    def job_file(job_id: str, name: str) -> FileResponse:
        if name not in job_files(job_id)["files"]:  # only names the store listed: no traversal
            raise HTTPException(status_code=404, detail="no such output file")
        tmp = Path(tempfile.mkdtemp(prefix="gpurelay-dl-"))
        target = tmp / Path(name).name
        try:
            ctx.storage.get_file(f"{final_prefix(find(job_id).id)}/{name}", target)
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(status_code=502, detail=f"download failed: {exc}") from exc
        return FileResponse(target, filename=target.name,
                            background=BackgroundTask(shutil.rmtree, tmp, True))  # fmt: skip

    def _analysis(saved: Any) -> Any:
        from gpurelay.analyzer import analyze_path

        try:
            return analyze_path(saved.path, requirements=saved.requirements)
        except SyntaxError as exc:
            raise FormError(f"{saved.path.name} has a syntax error (line {exc.lineno}): "
                            f"{exc.msg}") from exc  # fmt: skip

    def _analyze(form: UploadForm) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="gpurelay-up-") as tmp:
            saved = save_upload(form, Path(tmp))
            return {"lines": saved.lines, "analysis": _analysis(saved).to_json()}

    @app.post("/api/analyze")
    async def analyze_upload(
        request: Request, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> dict[str, Any]:
        require_action_header(x_gpurelay_action)
        try:
            form = parse_body(await request.body(), UploadForm)
            return await run_in_threadpool(_analyze, form)
        except FormError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _submit(form: SubmitForm) -> dict[str, Any]:
        specs = [spec.strip() for spec in form.data if spec.strip()]
        if any(spec.startswith("local:") for spec in specs):
            raise FormError("local: data refers to files on the server; use kaggle:, hf: or url:")
        try:
            data = [DataSpec.parse(spec) for spec in specs]
            estimate_h = parse_duration(form.estimate) / 3600 if form.estimate else None
        except ValueError as exc:
            raise FormError(str(exc)) from exc
        with tempfile.TemporaryDirectory(prefix="gpurelay-up-") as tmp:
            saved = save_upload(form, Path(tmp))
            analysis = _analysis(saved)
            if analysis.overall is ResumabilityLevel.RESTART_ONLY and not form.accept_restart_only:
                raise FormError("this job has no safe points: any interruption restarts it from "
                                "scratch. Tick 'Submit anyway' to accept that.")  # fmt: skip
            try:
                job = submit_job(
                    ctx, saved.path, name=(form.name or "").strip() or None,
                    priority=form.priority, requirements=saved.requirements, data=data,
                    estimate_h=estimate_h, prefer=form.prefer or None,
                    accept_restart_only=form.accept_restart_only,
                )  # fmt: skip
            except GpuRelayError as exc:
                raise FormError(str(exc)) from exc
        return {"ok": True, "job_id": job.id, "level": analysis.overall.value}

    @app.post("/api/submit")
    async def submit_upload(
        request: Request, x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER)
    ) -> dict[str, Any]:
        require_action_header(x_gpurelay_action)
        try:
            form = parse_body(await request.body(), SubmitForm)
            return await run_in_threadpool(_submit, form)
        except FormError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/quota/{backend}")
    async def quota_correction(
        backend: str,
        request: Request,
        x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER),
    ) -> dict[str, Any]:
        require_action_header(x_gpurelay_action)
        try:
            form = parse_body(await request.body(), QuotaForm)
            used = parse_duration(form.used, allow_zero=True) / 3600 if form.used else None
            remaining = (
                parse_duration(form.remaining, allow_zero=True) / 3600 if form.remaining else None
            )
            correct_quota(ctx, backend, remaining_h=remaining, used_h=used)
        except (FormError, ValueError, GpuRelayError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "message": f"recorded correction for {backend}"}

    @app.get("/api/settings")
    def settings() -> dict[str, Any]:
        from gpurelay.credentials import hf_token, kaggle_credentials_present

        cfg = ctx.config
        kaggle = cfg.backends.kaggle
        return {
            "version": __version__,
            "server": auth is not None,
            "viewer_access": auth is not None and auth.viewer_password is not None,
            "storage": {
                "kind": cfg.storage.kind,
                "repo": cfg.storage.hf_repo_id,
                "token": hf_token() is not None,
            },
            "backends": [
                {
                    "name": "kaggle",
                    "enabled": kaggle.enabled,
                    "account": kaggle.username,
                    "gpu": kaggle.accelerator,
                    "weekly_h": kaggle.weekly_gpu_hours,
                    "credentials": kaggle_credentials_present(),
                },
                {"name": "colab", "enabled": cfg.backends.colab_manual.enabled, "manual": True},
                {"name": "lightning", "enabled": cfg.backends.lightning.enabled},
                {"name": "local", "enabled": cfg.backends.local.enabled},
            ],
            "configured": list(controller.backends),
            "runtime": {
                "max_lost_work_s": cfg.runtime.max_lost_work_s,
                "heartbeat_s": cfg.runtime.heartbeat_s,
            },
        }

    @app.post("/api/jobs/{job_id}/{action}")
    def job_action(
        job_id: str,
        action: str,
        x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER),
    ) -> dict[str, Any]:
        require_action_header(x_gpurelay_action)
        if action not in ACTIONS:
            raise HTTPException(status_code=404, detail=f"unknown action {action!r}")
        find(job_id)
        try:
            target = request(ctx, job_id, action)
        except GpuRelayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "message": f"{action} requested for {target.id}"}

    return app
