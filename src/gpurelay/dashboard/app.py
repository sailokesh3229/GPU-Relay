"""``gpurelay dashboard``: local web UI (spec §15, plan Phase 8).

FastAPI plus one static page, no build step. It never runs the controller: it reads the state
database and storage, and records pause/resume/cancel requests that the running controller
applies (exactly like the CLI). Meant to be served on localhost only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import resources
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from gpurelay.controller.estimator import fit_message
from gpurelay.controller.lock import controller_running
from gpurelay.controller.service import Context, colab_session, merged_log, request, resolve_job
from gpurelay.controller.views import JobRow, backend_rows, budget_lines, job_rows
from gpurelay.errors import GpuRelayError
from gpurelay.models import Job, JobState, Segment

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
        "duration_s": end - seg.started_at.timestamp() if seg.started_at else 0.0,
        "start_checkpoint": seg.start_checkpoint_seq,
        "end_checkpoint": seg.end_checkpoint_seq,
        "message": seg.message,
    }


LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _host_name(header: str) -> str:
    """``Host`` header without the port (IPv6 literals keep their brackets)."""
    if header.startswith("["):
        return header.split("]", 1)[0] + "]"
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def create_app(
    ctx: Context,
    *,
    clock: Callable[[], float] = time.time,
    allowed_hosts: frozenset[str] = LOCAL_HOSTS,
) -> FastAPI:
    """``allowed_hosts``: accepted ``Host`` names. Anything else is refused, which stops DNS
    rebinding (a website pointing its own domain at 127.0.0.1 to reach this API)."""
    app = FastAPI(title="GPU Relay", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def check_host(request: Request, call_next: Any) -> Any:
        if _host_name(request.headers.get("host", "")) not in allowed_hosts:
            return JSONResponse({"detail": "unexpected Host header"}, status_code=421)
        return await call_next(request)

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
        return {
            "job": _job_json(row),
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

    @app.post("/api/jobs/{job_id}/{action}")
    def job_action(
        job_id: str,
        action: str,
        x_gpurelay_action: str | None = Header(default=None, alias=ACTION_HEADER),
    ) -> dict[str, Any]:
        if x_gpurelay_action != "1":
            raise HTTPException(status_code=403, detail=f"missing {ACTION_HEADER} header")
        if action not in ACTIONS:
            raise HTTPException(status_code=404, detail=f"unknown action {action!r}")
        find(job_id)
        try:
            target = request(ctx, job_id, action)
        except GpuRelayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "message": f"{action} requested for {target.id}"}

    return app
