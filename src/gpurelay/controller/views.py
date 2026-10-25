"""``gpurelay status`` rendering: jobs (progress, ETA) and backends (remaining budget)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from gpurelay.controller.daemon import Controller
from gpurelay.controller.estimator import BudgetLine, Estimate, estimate, fit_message
from gpurelay.durations import format_duration
from gpurelay.models import BudgetModel, Job, JobState, SegmentState

_STATE_COLORS = {
    JobState.COMPLETED: "green",
    JobState.RUNNING: "cyan",
    JobState.FAILED_USER: "red",
    JobState.FAILED_INFRA: "red",
    JobState.WAITING_FOR_QUOTA: "yellow",
    JobState.AWAITING_USER: "magenta",
    JobState.PAUSED: "yellow",
}


@dataclass
class JobRow:
    job: Job
    backend: str
    progress: str
    used_h: float
    eta_h: float | None
    eta_basis: str
    segments: int
    completed: int | None = None  # outermost loop progress, when known
    total: int | None = None


@dataclass
class BackendRow:
    name: str
    automated: bool
    gpu: str | None
    budget: str  # "unlimited", the budget model when unknown, or "30.0h weekly_rolling"
    budget_h: float | None
    used_h: float
    remaining: str
    remaining_h: float | None
    estimated: bool  # remaining is computed from a default, not corrected by the user
    next_reset: datetime | None
    running: int
    max_concurrent: int
    model: str = "weekly_rolling"
    unit: str = "h"  # what the platform grants: hours, usd or credits (P11)
    cost_per_h: float = 1.0
    speed: float = 1.0


def _fmt_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def job_rows(controller: Controller, now: float) -> list[JobRow]:
    rows = []
    for job in controller.db.list_jobs():
        segments = controller.db.segments(job.id)
        live = [
            s for s in segments if s.state in (SegmentState.RUNNING, SegmentState.AWAITING_USER)
        ]
        active = [s for s in live if s.role == "main"] or live
        workers = sum(1 for s in live if s.role == "worker")
        used = 0.0
        for seg in segments:
            if seg.started_at is None:
                continue
            end = seg.ended_at.timestamp() if seg.ended_at else now
            used += max(0.0, end - seg.started_at.timestamp()) / 3600
        status = controller._read_status(job.id) if active else None
        if status is not None and status.segment_id != active[0].id:
            status = None
        progress = "-"
        completed = total = None
        if status is not None and status.progress.loop:
            completed, total = next(iter(status.progress.loop.values()))
            progress = f"{completed}/{total}" if total is not None else f"{completed}"
        est = estimate(job, status, used)
        if job.state.is_terminal:
            est = Estimate(None, "done", used)
        rows.append(
            JobRow(
                job=job,
                backend=(active[0].backend + (f" +{workers} worker(s)" if workers else ""))
                if active
                else "-",
                progress=progress,
                used_h=used,
                eta_h=est.remaining_h,
                eta_basis=est.basis,
                segments=len(segments),
                completed=completed,
                total=total,
            )
        )
    return rows


def backend_rows(controller: Controller, now: float) -> list[BackendRow]:
    active = controller.db.active_segments()
    rows = []
    for slot in controller.backends.values():
        quota = controller.quota_state(slot, now)
        estimated = False
        if slot.budget.unlimited:
            budget, remaining = "unlimited", "unlimited"
        elif slot.budget.hours is None:
            budget, remaining = slot.budget.model.value, "unknown"
        else:
            budget = f"{slot.budget.hours:.1f}h {slot.budget.model.value}"
            remaining = f"{quota.remaining_h:.2f}h" if quota.remaining_h is not None else "unknown"
            if slot.budget.estimate and not quota.corrected:
                remaining += " (est.)"
                estimated = True
        rows.append(
            BackendRow(
                name=slot.name,
                automated=slot.automated,
                gpu=slot.gpu_label,
                budget=budget,
                budget_h=slot.budget.hours,
                used_h=quota.used_h,
                remaining=remaining,
                remaining_h=quota.remaining_h,
                estimated=estimated,
                next_reset=quota.next_reset,
                running=sum(1 for s in active if s.backend == slot.name),
                max_concurrent=slot.max_concurrent,
                model=slot.budget.model.value,
                unit=slot.budget.unit,
                cost_per_h=slot.budget.cost_per_h,
                speed=slot.speed,
            )
        )
    return rows


def budget_lines(controller: Controller, now: float) -> list[BudgetLine]:
    lines = []
    for slot in controller.backends.values():
        quota = controller.quota_state(slot, now)
        weekly = slot.budget.hours
        if weekly is not None and slot.budget.model is BudgetModel.MONTHLY:
            weekly = weekly * 7 / 30
        elif weekly is not None and slot.budget.model is BudgetModel.DAILY:
            weekly = weekly * 7
        lines.append(
            BudgetLine(slot.name, quota.remaining_h, weekly, slot.automated, slot.budget.unlimited)
        )
    return lines


def render_status(console: Console, controller: Controller, now: float) -> None:
    jobs = Table(title="Jobs", show_header=True, header_style="bold")
    for column in ("job", "name", "state", "backend", "progress", "GPU used", "ETA", "segs"):
        jobs.add_column(column, no_wrap=column in ("job", "state"))
    jobs.add_column("message")
    for row in job_rows(controller, now):
        color = _STATE_COLORS.get(row.job.state, "white")
        eta = "-" if row.eta_h is None else f"~{row.eta_h:.2f}h ({row.eta_basis})"
        jobs.add_row(
            row.job.id,
            row.job.name,
            f"[{color}]{row.job.state.value}[/{color}]",
            row.backend,
            row.progress,
            format_duration(row.used_h * 3600),
            eta,
            str(row.segments),
            row.job.message,
        )
    console.print(jobs)

    backends = Table(title="GPU budget", show_header=True, header_style="bold")
    for column in (
        "backend",
        "type",
        "GPU",
        "budget",
        "used (window)",
        "remaining",
        "next reset",
        "running",
    ):
        backends.add_column(column)
    for b in backend_rows(controller, now):
        backends.add_row(
            b.name,
            "automated" if b.automated else "manual",
            b.gpu or "-",
            b.budget,
            f"{b.used_h:.2f}h",
            b.remaining,
            _fmt_time(b.next_reset),
            f"{b.running}/{b.max_concurrent}",
        )
    console.print(backends)


def render_job(console: Console, controller: Controller, job: Job, now: float) -> None:
    row = next(r for r in job_rows(controller, now) if r.job.id == job.id)
    console.print(f"[bold]{job.id}[/bold] {job.name} - {job.state.value}  {job.message}")
    used = format_duration(row.used_h * 3600)
    console.print(f"progress {row.progress}, {used} GPU time used, {row.segments} segment(s)")
    if not job.state.is_terminal:
        console.print(fit_message(row.eta_h, budget_lines(controller, now)))
    table = Table(show_header=True, header_style="bold")
    for column in ("#", "segment", "backend", "GPU", "state", "duration", "checkpoints", "message"):
        table.add_column(column)
    for seg in controller.db.segments(job.id):
        end = seg.ended_at.timestamp() if seg.ended_at else now
        duration = end - seg.started_at.timestamp() if seg.started_at else 0.0
        table.add_row(
            str(seg.index),
            seg.id,
            seg.backend,
            seg.gpu_model or "-",
            seg.state.value,
            format_duration(duration),
            f"{seg.start_checkpoint_seq} -> {seg.end_checkpoint_seq}",
            seg.message,
        )
    console.print(table)
