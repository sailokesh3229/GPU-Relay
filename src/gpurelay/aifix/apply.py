"""Apply and rerun (spec §20.1, plan A2): a fixed copy of a failed job, as a new attempt.

The new job is the old one's bundle with the changed cells (or script) swapped in: the same
other files, requirements, data (as already staged), priority, platform choice and paid
limit. It starts from the beginning: the failed attempt's checkpoints belong to the old code.
The failed job stays as it was, and the two are linked (``Job.retry_of``, ``Job.fix_id``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from gpurelay.aifix.request import FixError
from gpurelay.models import Job, JobState


def fixed_entry(entry: str, source: bytes, changes: list[dict[str, Any]]) -> bytes:
    """``source`` with each change's new code in place (code cells counted from 1)."""
    if not entry.endswith(".ipynb"):
        [change] = changes if len(changes) == 1 else [None]  # a script is one cell
        if change is None or int(change["cell"]) != 1:
            raise FixError("a script fix must change the script as one piece")
        return str(change["after"]).encode("utf-8")
    import nbformat

    notebook = nbformat.reads(source.decode("utf-8"), 4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for change in changes:
        number = int(change["cell"])
        if not 1 <= number <= len(code):
            raise FixError(f"the fix changes cell {number}, which the notebook does not have")
        code[number - 1].source = str(change["after"])
    for cell in code:  # outputs of the failed run would be misleading next to new code
        cell.outputs, cell.execution_count = [], None
    text: str = nbformat.writes(notebook)
    return text.encode("utf-8")


def root_of(ctx: Any, job: Job) -> Job:
    """The first attempt of a chain of reruns."""
    seen = {job.id}
    while job.retry_of and (parent := ctx.db.get_job(job.retry_of)) is not None:
        if parent.id in seen:
            break
        seen.add(parent.id)
        job = parent
    return job


def chain(ctx: Any, job: Job) -> list[Job]:
    """Every attempt of ``job``'s chain, first to last."""
    root = root_of(ctx, job)
    jobs, current = [root], root
    children: dict[str, Job] = {j.retry_of: j for j in ctx.db.list_jobs() if j.retry_of}
    while current.id in children and len(jobs) < 50:
        current = children[current.id]
        jobs.append(current)
    return jobs


def rerun_of(ctx: Any, job: Job) -> Job | None:
    return next((j for j in ctx.db.list_jobs() if j.retry_of == job.id), None)


def resubmit(ctx: Any, job: Job, fix: dict[str, Any]) -> Job:
    """Submit ``job``'s code with ``fix`` applied; returns the new job."""
    from gpurelay.bundle.packager import read_bundle_files
    from gpurelay.controller.service import submit_job

    if job.state is not JobState.FAILED_USER:
        raise FixError("only a job that stopped with an error in its code can be rerun with a fix")
    result = fix.get("result") or {}
    changes = result.get("changes") or []
    if fix.get("job_id") != job.id or not result.get("ok") or not changes:
        raise FixError("this fix has no change to apply")
    if fix.get("applied_job"):
        raise FixError(f"this fix was applied already (job {fix['applied_job']})")
    existing = rerun_of(ctx, job)
    if existing is not None:
        raise FixError(f"this job was rerun already as {existing.id}")
    try:
        manifest, files = read_bundle_files(ctx.storage, job.id)
    except Exception as exc:
        raise FixError(f"the job's files could not be read from storage: {exc}") from exc
    files = {**files, manifest.entry: fixed_entry(manifest.entry, files[manifest.entry], changes)}
    attempts = len(chain(ctx, job))
    base = root_of(ctx, job).name
    with tempfile.TemporaryDirectory(prefix="gpurelay-rerun-") as tmp:
        root = Path(tmp).resolve()
        for rel, data in files.items():
            target = root / rel
            if root not in target.resolve().parents:
                raise FixError(f"unexpected file name in the bundle: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        requirements = root / manifest.requirements if manifest.requirements else None
        own = (manifest.entry, manifest.requirements)
        extra = {rel: root / rel for rel in files if rel not in own}
        new = submit_job(
            ctx, root / manifest.entry, name=f"{base} · fix {attempts}", priority=job.priority,
            requirements=requirements, data=[s.spec for s in manifest.data],
            staged=list(manifest.data), estimate_h=job.estimate_h, prefer=job.prefer_backend,
            accept_restart_only=job.accept_restart_only, extra_files=extra,
            max_paid_usd=job.max_paid_usd, retry_of=job.id, fix_id=str(fix["id"]),
        )  # fmt: skip
    import time

    ctx.db.add_event("ai_fix_rerun", f"rerun with an AI fix as {new.id}", now=time.time(),
                     job_id=job.id)  # fmt: skip
    return new
