""" "Fix it with AI" on the job page (spec §20, plan A1-A3).

``mode``: where the AI tools run.

* ``"local"``: on this computer (the desktop app, ``gpurelay dashboard``): the web app asks
  Claude Code or Codex itself, one request at a time, in a background thread;
* ``"helper"``: on the user's laptop, while this GPU Relay runs on the website: requests wait
  until the user's helper collects them (``/api/helper/*``, reached only through the website
  with the user's helper key) and posts the answers back.

A fix the user approves is applied as a new attempt of the job (A2, ``aifix/apply``). With
automatic reruns on (``[aifix] auto``), a job that fails in its code gets a fix request and,
when the AI proposes a change, a rerun with it, up to ``max_attempts`` fixes and
``max_gpu_hours`` for all attempts together. When a rerun ends, GPU Relay records whether the
fix worked, which decides the model asked first next time (A3, ``aifix/choice``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from gpurelay.aifix import choice
from gpurelay.aifix.apply import chain, rerun_of, resubmit
from gpurelay.aifix.board import DONE, FAILED, WAITING, WORKING, FixBoard, public
from gpurelay.aifix.request import LOG_TAIL_CHARS, FixError, FixRequest, code_cells
from gpurelay.aifix.tool import ToolInfo
from gpurelay.errors import GpuRelayError
from gpurelay.models import Job, JobState
from gpurelay.runtime.errorloc import parse_location

log = logging.getLogger(__name__)

MODES = ("local", "helper")
DETECT_CACHE_S = 60.0
MAX_ANSWER_BYTES = 2 * 1024 * 1024
TICK_S = 10.0
AUTO_WINDOW_S = 24 * 3600.0  # automatic reruns only for jobs that failed this recently
WAITING_BUSY_S = 30 * 60.0  # a request waiting this long no longer keeps GPU Relay awake


class AiSettingsForm(BaseModel):
    auto: bool = False
    max_attempts: int = Field(default=2, ge=1, le=5)
    max_gpu_hours: float = Field(default=6.0, gt=0, le=200)


def fix_request(ctx: Any, job: Job) -> FixRequest:
    """What is sent for ``job``: its code, the error with its location, the log's end."""
    from gpurelay.bundle.packager import read_bundle_file, read_manifest
    from gpurelay.controller.service import merged_log

    try:
        manifest = read_manifest(ctx.storage, job.id)
        cells = code_cells(read_bundle_file(ctx.storage, job.id, manifest.entry), manifest.entry)
    except Exception as exc:
        raise FixError(f"the job's code could not be read from storage: {exc}") from exc
    try:
        log_tail = merged_log(ctx, job)[-LOG_TAIL_CHARS:]
    except Exception:
        log_tail = ""  # the error message alone still says what failed
    where = parse_location(job.message)
    segments = ctx.db.segments(job.id)
    return FixRequest(job_id=job.id, name=job.name, entry=manifest.entry, cells=cells,
                      error=job.message, cell=where[0] if where else None,
                      line=where[1] if where else None, log_tail=log_tail,
                      platform=segments[-1].backend if segments else "")  # fmt: skip


def outcome(original_error: str, rerun: Job) -> str | None:
    """Whether a rerun shows its fix worked: "fixed" (it finished, or got past that error),
    "not_fixed" (the same error again), "unknown" (stopped for another reason); None while
    it still runs."""
    if not rerun.state.is_terminal:
        return None
    if rerun.state is JobState.COMPLETED:
        return "fixed"
    if rerun.state is not JobState.FAILED_USER:
        return "unknown"
    same_place = parse_location(rerun.message) == parse_location(original_error)
    same_kind = choice.error_kind(rerun.message) == choice.error_kind(original_error)
    return "not_fixed" if same_place and same_kind else "fixed"


def gpu_hours(ctx: Any, jobs: list[Job], now: float) -> float:
    total = 0.0
    for job in jobs:
        for seg in ctx.db.segments(job.id):
            if seg.started_at is not None:
                end = seg.ended_at.timestamp() if seg.ended_at else now
                total += max(0.0, end - seg.started_at.timestamp())
    return total / 3600.0


class AiFix:
    def __init__(
        self,
        ctx: Any,
        find: Callable[[str], Job],
        mode: str,
        *,
        action_header: str,
        board: FixBoard | None = None,
        solver: Callable[..., dict[str, Any]] = choice.solve,
        detect: Callable[[], list[ToolInfo]] = choice.detect_all,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        save_settings: Callable[[Any], None] | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"AI fix mode must be one of {MODES}")
        self.ctx, self.find, self.mode, self.action_header = ctx, find, mode, action_header
        self.board = board or FixBoard(ctx.home / "aifix")
        self.stats = choice.Stats(ctx.home / "aifix" / "stats.json")
        self.solver, self.detect, self.clock, self.wall = solver, detect, clock, wall
        self.save_settings = save_settings or self._save_config
        self._lock = threading.Lock()
        self._worker = False
        self._ticking = False
        self._last_tick: float | None = None
        self._tools: list[dict[str, Any]] | None = None
        self._checked: float | None = None
        self._pending = False
        self.router = self._routes()

    # ---------------------------------------------------------------- settings and tools

    @property
    def settings(self) -> Any:
        return self.ctx.config.aifix

    def _save_config(self, settings: Any) -> None:
        from gpurelay.config import save_config
        from gpurelay.constants import CONFIG_FILENAME

        self.ctx.config.aifix = settings
        save_config(self.ctx.config, self.ctx.home / CONFIG_FILENAME)

    def tools(self) -> list[dict[str, Any]]:
        if self.mode == "helper":
            return list(self.board.helper().get("tools", []))
        with self._lock:
            if self._checked is None or self.clock() - self._checked > DETECT_CACHE_S:
                self._tools = [t.to_json() for t in self.detect()]
                self._checked = self.clock()
            return list(self._tools or [])

    # ---------------------------------------------------------------- requests

    def create_fix(self, job: Job, *, auto: bool = False) -> dict[str, Any]:
        """A fix request for a job that failed in its code. Earlier fixes of the same chain
        of attempts (they did not work) move the plan on to stronger models."""
        if job.state is not JobState.FAILED_USER:
            raise FixError("only a job that stopped with an error in its code can be fixed")
        req = fix_request(self.ctx, job)
        req.check_size()
        attempts = chain(self.ctx, job)
        earlier = len(self.board.for_jobs({j.id for j in attempts}))
        steps = choice.plan(req, self.settings, self.tools(), self.stats, earlier)
        fix = self.board.create(req, plan=steps, auto=auto)
        if self.mode == "local":
            self._start_worker()
        return fix

    def _work(self) -> None:
        try:
            while (fix := self.board.claim()) is not None:
                available = {t["name"] for t in self.tools() if t.get("signed_in")}
                req = FixRequest.from_json(fix["request"])
                try:
                    result = self.solver(req, fix.get("plan") or [], available)
                except Exception as exc:  # never leave a request "working"
                    result = {"ok": False, "error": f"the AI tool failed: {exc}"}
                self.board.finish(fix["id"], result)
        finally:
            with self._lock:
                self._worker = False

    def _start_worker(self) -> None:
        with self._lock:
            if self._worker:
                return
            self._worker = True
        threading.Thread(target=self._work, name="gpurelay-aifix", daemon=True).start()

    def apply(self, job: Job, fix_id: str) -> Job:
        fix = self.board.get(fix_id)
        if fix is None or fix["job_id"] != job.id:
            raise FixError("no such fix for this job")
        try:
            new = resubmit(self.ctx, job, fix)
        except GpuRelayError as exc:
            raise FixError(str(exc)) from exc
        self.board.mark_applied(fix_id, new.id)
        return new

    # ---------------------------------------------------------------- the regular look

    def maybe_tick(self) -> None:
        """At most every ``TICK_S``, in the background (called on each overview request:
        the page asks every 10 s, the website every 30 s)."""
        with self._lock:
            now = self.clock()
            if self._ticking or (self._last_tick is not None and now - self._last_tick < TICK_S):
                return
            self._ticking, self._last_tick = True, now

        def run() -> None:
            try:
                self.tick()
            except Exception as exc:  # one bad look must not stop the next
                log.warning("AI fix check failed: %s", exc)
            finally:
                with self._lock:
                    self._ticking = False

        threading.Thread(target=run, name="gpurelay-aifix-tick", daemon=True).start()

    def tick(self) -> None:
        """Record what finished reruns showed; with automatic reruns on, ask for fixes and
        apply them within the limits."""
        db = self.ctx.db
        for fix in self.board.all():
            if not fix.get("applied_job") or fix.get("outcome"):
                continue
            rerun = db.get_job(fix["applied_job"])
            result = outcome(fix["request"]["error"], rerun) if rerun is not None else "unknown"
            if result is None:
                continue
            self.board.set_outcome(fix["id"], result)
            step = (fix.get("result") or {}).get("step") or {}
            if result in ("fixed", "not_fixed") and step.get("tool"):
                kind = choice.error_kind(fix["request"]["error"])
                self.stats.record(kind, step["tool"], str(step.get("model") or "default"),
                                  result == "fixed")  # fmt: skip
        pending = False
        if self.settings.auto:
            for job in db.list_jobs([JobState.FAILED_USER]):
                pending = self._auto(job) or pending
        self._pending = pending

    def _auto(self, job: Job) -> bool:
        """One step of automatic reruns for a failed job; True while it still has work."""
        settings, now = self.settings, self.wall()
        failed_at = job.updated_at.replace(tzinfo=job.updated_at.tzinfo or timezone.utc)
        if now - failed_at.timestamp() > AUTO_WINDOW_S or rerun_of(self.ctx, job) is not None:
            return False
        latest = self.board.latest(job.id)
        if latest is None:
            attempts = chain(self.ctx, job)
            if len(attempts) - 1 >= settings.max_attempts:
                return False
            if gpu_hours(self.ctx, attempts, now) >= settings.max_gpu_hours:
                return False
            try:
                self.create_fix(job, auto=True)
            except FixError as exc:
                log.info("no automatic fix for %s: %s", job.id, exc)
                return False
            return True
        if latest["state"] in (WAITING, WORKING):
            return True
        result = latest.get("result") or {}
        if latest.get("auto") and latest["state"] == DONE and result.get("changes"):
            try:
                self.apply(job, latest["id"])
            except FixError as exc:
                log.info("automatic rerun of %s not made: %s", job.id, exc)
        return False

    def busy(self) -> bool:
        """Work in hand: a request not answered yet, or an automatic rerun still to make (the
        website keeps this user's GPU Relay running meanwhile)."""
        now = self.wall()
        for fix in self.board.all():
            if fix["state"] == WORKING or (
                fix["state"] == WAITING and now - float(fix["created_at"]) < WAITING_BUSY_S
            ):
                return True
        return self._pending

    # ---------------------------------------------------------------- routes

    def _require(self, value: str | None) -> None:
        if value != "1":  # a custom header: a form on another site cannot send it
            raise HTTPException(status_code=403, detail=f"missing {self.action_header} header")

    def _routes(self) -> APIRouter:
        router = APIRouter()
        header = self.action_header

        @router.get("/api/ai")
        def ai_status() -> dict[str, Any]:
            tools = self.tools()
            helper = self.board.helper() if self.mode == "helper" else None
            connected = helper is None or helper["connected"]
            s = self.settings
            return {
                "mode": self.mode,
                "ready": connected and any(t.get("signed_in") for t in tools),
                "tools": tools,
                "helper": helper,
                "settings": {
                    "auto": s.auto,
                    "max_attempts": s.max_attempts,
                    "max_gpu_hours": s.max_gpu_hours,
                },
                # On the website the settings live with the account (Set up, step 6).
                "settings_here": self.mode == "local",
            }

        @router.post("/api/ai/settings")
        def ai_settings(
            form: AiSettingsForm, x_gpurelay_action: str | None = Header(default=None, alias=header)
        ) -> dict[str, Any]:
            self._require(x_gpurelay_action)
            if self.mode != "local":
                raise HTTPException(409, detail="change this under Set up accounts → AI auto-fix")
            settings = self.settings.model_copy(update=form.model_dump())
            self.save_settings(settings)
            return {"ok": True}

        @router.get("/api/jobs/{job_id}/fix")
        def latest_fix(job_id: str) -> dict[str, Any]:
            job = self.find(job_id)
            return {"fix": public(self.board.latest(job.id)), "attempt": self._attempt(job)}

        @router.post("/api/jobs/{job_id}/fix")
        async def ask_fix(
            job_id: str, x_gpurelay_action: str | None = Header(default=None, alias=header)
        ) -> dict[str, Any]:
            self._require(x_gpurelay_action)
            job = self.find(job_id)
            if job.state is not JobState.FAILED_USER:
                raise HTTPException(409, detail="only a job that stopped with an error in its "
                                    "code can be fixed")  # fmt: skip
            if self.board.busy(job.id):
                return {"fix": public(self.board.latest(job.id))}
            try:
                fix = await run_in_threadpool(self.create_fix, job)
            except FixError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"fix": public(fix)}

        @router.post("/api/jobs/{job_id}/fix/{fix_id}/apply")
        async def apply_fix(
            job_id: str,
            fix_id: str,
            x_gpurelay_action: str | None = Header(default=None, alias=header),
        ) -> dict[str, Any]:
            self._require(x_gpurelay_action)
            job = self.find(job_id)
            try:
                new = await run_in_threadpool(self.apply, job, fix_id)
            except FixError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"ok": True, "job_id": new.id, "name": new.name}

        if self.mode == "helper":

            @router.post("/api/helper/poll")
            async def helper_poll(
                request: Request, x_gpurelay_action: str | None = Header(default=None, alias=header)
            ) -> dict[str, Any]:
                """The helper says it is there (and which AI tools it has); it gets the next
                request to answer, if any."""
                self._require(x_gpurelay_action)
                body = await _json(request)
                self.board.seen(body.get("tools"), str(body.get("version", "")))
                fix = self.board.claim()
                if fix is None:
                    return {"fix": None}
                return {"fix": {"id": fix["id"], "model": fix.get("model"),
                                "plan": fix.get("plan"), "request": fix["request"]}}  # fmt: skip

            @router.post("/api/helper/fixes/{fix_id}")
            async def helper_answer(
                fix_id: str,
                request: Request,
                x_gpurelay_action: str | None = Header(default=None, alias=header),
            ) -> dict[str, Any]:
                self._require(x_gpurelay_action)
                fix = self.board.finish(fix_id, await _json(request))
                if fix is None:
                    raise HTTPException(404, detail="no such request waiting for an answer")
                return {"ok": True, "state": fix["state"]}

        return router

    def _attempt(self, job: Job) -> dict[str, Any]:
        """Where the job stands in its chain of attempts (for the job page)."""
        attempts = chain(self.ctx, job)
        index = next((i for i, j in enumerate(attempts) if j.id == job.id), 0)
        rerun = attempts[index + 1] if index + 1 < len(attempts) else None
        return {
            "number": index + 1,
            "of": len(attempts),
            "first": attempts[0].id,
            "previous": attempts[index - 1].id if index > 0 else None,
            "rerun": None
            if rerun is None
            else {"id": rerun.id, "name": rerun.name, "state": rerun.state.value},
        }


async def _json(request: Request) -> dict[str, Any]:
    import json

    raw = await request.body()
    if len(raw) > MAX_ANSWER_BYTES:
        raise HTTPException(status_code=413, detail="the answer is too large")
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="send JSON") from None
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="send a JSON object")
    return data


__all__ = ["DONE", "FAILED", "AiFix", "fix_request", "outcome"]
