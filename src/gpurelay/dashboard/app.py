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
from gpurelay.constants import COLAB_LAUNCHER_URL
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
from gpurelay.controller.spend import segment_cost
from gpurelay.controller.views import JobRow, backend_rows, budget_lines, job_rows
from gpurelay.dashboard.auth import COOKIE_NAME, OWNER, VIEWER, Auth
from gpurelay.dashboard.desktop_api import DesktopHooks, SetupDeps, add_launch_key, desktop_router
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


RESULT_TEXT_MAX = 20_000  # characters of one output shown on the page (the file has it all)
RESULT_IMAGE_MAX = 3 * 1024 * 1024  # base64 characters of one image shown on the page


ERROR_CONTEXT = 6  # lines shown before and after the failing line


def error_snippet(source: bytes, entry: str, cell: int | None, line: int) -> dict[str, Any] | None:
    """The code around a failing line: ``cell`` counts code cells from 1 (notebooks)."""
    text = source.decode("utf-8", "replace")
    if entry.endswith(".ipynb"):
        import nbformat

        code = [c.source for c in nbformat.reads(text, 4).cells if c.cell_type == "code"]
        if cell is None or not 1 <= cell <= len(code):
            return None
        text = code[cell - 1]
    lines = text.splitlines()
    if not 1 <= line <= len(lines):
        return None
    first = max(1, line - ERROR_CONTEXT)
    last = min(len(lines), line + ERROR_CONTEXT)
    return {
        "cell": cell,
        "line": line,
        "first": first,
        "lines": lines[first - 1 : last],
    }


def notebook_results(path: Path) -> list[dict[str, Any]]:
    """The code cells of an output notebook with their outputs, ready for the job page."""
    import nbformat

    def clip(text: str) -> str:
        return text if len(text) <= RESULT_TEXT_MAX else text[:RESULT_TEXT_MAX] + "\n… (cut)"

    cells = []
    for cell in nbformat.read(str(path), as_version=4).cells:
        if cell.cell_type != "code":
            continue
        outputs: list[dict[str, str]] = []
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                outputs.append({"kind": "text", "text": clip(str(out.get("text", "")))})
            elif kind == "error":
                trace = "\n".join(out.get("traceback", []))
                trace = trace or f"{out.get('ename')}: {out.get('evalue')}"
                outputs.append({"kind": "error", "text": clip(trace)})
            elif kind in ("execute_result", "display_data"):
                data = out.get("data", {})
                png = data.get("image/png")
                if isinstance(png, str) and len(png) <= RESULT_IMAGE_MAX:
                    outputs.append({"kind": "image", "png": png.replace("\n", "")})
                elif "text/plain" in data:
                    outputs.append({"kind": "text", "text": clip(str(data["text/plain"]))})
        cells.append({"source": clip(str(cell.source)), "outputs": outputs})
    return cells


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
        "max_paid_usd": job.max_paid_usd,
        "cost_usd": round(row.cost_usd, 4),
        "retry_of": job.retry_of,
        "fix_id": job.fix_id,
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
        "paid_usd_per_h": seg.paid_usd_per_h,
        "cost_usd": round(segment_cost(seg, now), 4),
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
    desktop: DesktopHooks | None = None,
    launch_key: str | None = None,
    setup_deps: SetupDeps | None = None,
    ai_fix: str = "local",
    ai: Any = None,
    platforms_url: str | None = None,
) -> FastAPI:
    """``allowed_hosts``: accepted ``Host`` names. Anything else is refused, which stops DNS
    rebinding (a website pointing its own domain at 127.0.0.1 to reach this API). None accepts
    any host, for a server behind ``auth``.

    ``auth``: require a login; viewers may only read. Without it everyone is the owner.

    ``ai_fix``: where "Fix it with AI" runs the user's AI tool (``dashboard/aifix_routes``):
    ``"local"`` on this computer, ``"helper"`` on the user's laptop (the website).
    ``ai``: a ready-made :class:`~gpurelay.dashboard.aifix_routes.AiFix` instead (tests).
    ``platforms_url``: the "Your GPU platforms" page (spec §21.1) for Settings: the
    website's (``/platforms``); the desktop app serves its own."""
    if desktop is not None and (auth is not None or launch_key is None):
        raise ValueError("the desktop app runs locally with a launch key, never behind a login")
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
        request.state.anonymous = role is None and auth.public_view
        if request.state.anonymous:
            role = VIEWER  # public view: look without logging in, never act
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
        return {
            "role": request.state.role,
            "login": auth is not None,
            "anonymous": getattr(request.state, "anonymous", False),
            "desktop": desktop is not None,
            "configured": True,
            "platforms_url": "platforms" if desktop is not None else platforms_url,
        }

    controller = ctx.controller

    def find(job_id: str) -> Job:
        try:
            return resolve_job(ctx, job_id)
        except GpuRelayError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def row_for(job: Job, now: float) -> JobRow:
        return next(r for r in job_rows(controller, now) if r.job.id == job.id)

    if ai is None:
        from gpurelay.dashboard.aifix_routes import AiFix

        ai = AiFix(ctx, find, ai_fix, action_header=ACTION_HEADER)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("gpurelay.dashboard").joinpath("index.html").read_text("utf-8")

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        now = clock()
        rows = sorted(job_rows(controller, now), key=lambda r: r.job.submitted_at, reverse=True)
        ai.maybe_tick()  # automatic reruns and the record of fixes (spec §20)
        return {
            "now": _iso(now),
            "controller_running": controller_running(ctx.home),
            "ai_busy": ai.busy(),
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
                    "model": b.model,
                    "unit": b.unit,
                    "cost_per_h": b.cost_per_h,
                    "speed": b.speed,
                    "hold": b.hold,
                    "paid_usd_per_h": b.paid_usd_per_h,
                    "paid_cap_usd": b.paid_cap_usd,
                    "paid_spent_usd": round(b.paid_spent_usd, 4),
                    "paid_only": b.paid_only,
                    "credit_left_usd": (
                        None if b.credit_left_usd is None else round(b.credit_left_usd, 2)
                    ),
                    "credit_source": b.credit_source,
                    "credit_as_of": _iso(b.credit_as_of),
                    "use_credits": b.use_credits,
                    "daily_h": daily_usage(ctx.db.ledger(b.name), now),
                }
                for b in backend_rows(controller, now)
            ],
            "colab_actions": [
                {
                    "job_id": r.job.id,
                    "name": r.job.name,
                    "command": f"gpurelay colab {r.job.id}",
                    "open_url": COLAB_LAUNCHER_URL,
                }
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

    @app.get("/api/jobs/{job_id}/error")
    def job_error(job_id: str) -> dict[str, Any]:
        """Where the job's code failed, with the code around that line (job page)."""
        from gpurelay.bundle.packager import read_bundle_file, read_manifest
        from gpurelay.runtime.errorloc import parse_location

        job = find(job_id)
        where = parse_location(job.message) if job.state is JobState.FAILED_USER else None
        if where is None:
            return {"found": False}
        try:
            manifest = read_manifest(ctx.storage, job.id)
            source = read_bundle_file(ctx.storage, job.id, manifest.entry)
        except Exception:
            return {"found": False}
        snippet = error_snippet(source, manifest.entry, where[0], where[1])
        return {"found": snippet is not None, "entry": manifest.entry, **(snippet or {})}

    @app.get("/api/jobs/{job_id}/results")
    def job_results(job_id: str) -> dict[str, Any]:
        """What a completed job produced, to show on its page: the figures, and for a
        notebook every cell's outputs (text, images, errors) as in Colab."""
        listing = job_files(job_id)
        if not listing["completed"]:
            return {"completed": False}
        files = listing["files"]
        figures = [f for f in files if f.startswith("figures/") and f.endswith(".png")]
        cells: list[dict[str, Any]] | None = None
        if "output.ipynb" in files:
            with tempfile.TemporaryDirectory(prefix="gpurelay-results-") as tmp:
                target = Path(tmp) / "output.ipynb"
                try:
                    ctx.storage.get_file(f"{final_prefix(find(job_id).id)}/output.ipynb", target)
                    cells = notebook_results(target)
                except Exception as exc:
                    raise HTTPException(502, detail=f"output notebook unavailable: {exc}") from exc
        others = [f for f in files if f not in figures and f not in ("output.ipynb",)]
        return {"completed": True, "figures": figures, "cells": cells, "files": others}

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
                    max_paid_usd=form.max_paid_usd,
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
            slot = controller.backends.get(backend)
            if form.used_amount is not None or form.remaining_amount is not None:
                if slot is None:
                    raise FormError(f"unknown backend {backend!r}")
                cost = slot.budget.cost_per_h  # dollars or credits per GPU hour
                if form.used_amount is not None:
                    used = form.used_amount / cost
                if form.remaining_amount is not None:
                    remaining = form.remaining_amount / cost
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
            "public_view": auth is not None and auth.public_view,
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
                {
                    "name": "modal",
                    "enabled": cfg.backends.modal.enabled,
                    "gpu": cfg.backends.modal.gpu,
                    "monthly_usd": cfg.backends.modal.monthly_credit_usd,
                },
                {"name": "local", "enabled": cfg.backends.local.enabled},
            ],
            "configured": list(controller.backends),
            "runtime": {
                "max_lost_work_s": cfg.runtime.max_lost_work_s,
                "heartbeat_s": cfg.runtime.heartbeat_s,
            },
        }

    app.include_router(ai.router)  # before the job actions: /api/jobs/{id}/fix is its own

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

    if desktop is not None:
        app.include_router(desktop_router(desktop, setup_deps))
    if launch_key is not None:
        add_launch_key(app, launch_key)
    return app
