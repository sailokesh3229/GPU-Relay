"""The fix requests of one user's GPU Relay, and the helper that answers them (spec §20).

A request is made on the job page ("Fix it with AI") and waits here until the AI tool answers:
on the user's own computer the web app asks it directly; on the website the user's helper
(``gpurelay helper``) collects it, asks the AI on their laptop and posts the answer back. Each
request is a small JSON file in the GPU Relay home (``aifix/<id>.json``). An answer is checked
again here against the request's own code, whoever sent it.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpurelay.aifix.request import FixError, FixRequest, changes_from

WAITING, WORKING, DONE, FAILED = "waiting", "working", "done", "failed"
HELPER_SEEN_S = 30.0  # a helper that asked within this long is connected
WORK_TIMEOUT_S = 15 * 60.0  # a collected request with no answer by then has failed
KEEP = 50  # requests kept per user (older ones are removed)
MAX_TOOLS = 8


class FixBoard:
    def __init__(self, root: Path, clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root)
        self.clock = clock
        self._lock = threading.RLock()
        self._helper: dict[str, Any] | None = None

    # ---------------------------------------------------------------- files

    def _path(self, fix_id: str) -> Path:
        return self.root / f"{fix_id}.json"

    def _save(self, fix: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(fix["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(fix), encoding="utf-8")
        tmp.replace(self._path(fix["id"]))

    def _all(self) -> list[dict[str, Any]]:
        fixes = []
        for path in self.root.glob("fix-*.json") if self.root.is_dir() else []:
            try:
                fixes.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        fixes.sort(key=lambda f: float(f.get("created_at", 0)))
        now = self.clock()
        for fix in fixes:  # collected but never answered (the helper stopped)
            if fix["state"] == WORKING and now - float(fix["updated_at"]) > WORK_TIMEOUT_S:
                self._end(fix, {"ok": False, "error": "the AI did not answer in time; try again"})
        return fixes

    def _prune(self) -> None:
        for fix in self._all()[:-KEEP]:
            self._path(fix["id"]).unlink(missing_ok=True)

    # ---------------------------------------------------------------- requests

    def create(
        self,
        req: FixRequest,
        model: str | None = None,
        *,
        plan: list[dict[str, Any]] | None = None,
        auto: bool = False,
    ) -> dict[str, Any]:
        """A new request. ``plan``: the tools and models to try in order (``aifix/choice``);
        ``auto``: made by automatic reruns, applied without asking when it has a fix."""
        now = self.clock()
        steps = plan or [{"tool": "claude_code", "model": model}]
        fix = {"id": "fix-" + secrets.token_hex(6), "job_id": req.job_id, "state": WAITING,
               "created_at": now, "updated_at": now, "model": steps[0].get("model"),
               "plan": steps, "auto": auto, "applied_job": None, "outcome": None,
               "request": req.to_json(), "result": None}  # fmt: skip
        with self._lock:
            self._save(fix)
            self._prune()
        return fix

    def get(self, fix_id: str) -> dict[str, Any] | None:
        with self._lock:
            return next((f for f in self._all() if f["id"] == fix_id), None)

    def latest(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            mine = [f for f in self._all() if f["job_id"] == job_id]
            return mine[-1] if mine else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._all()

    def for_jobs(self, job_ids: set[str]) -> list[dict[str, Any]]:
        return [f for f in self.all() if f["job_id"] in job_ids]

    def _update(self, fix_id: str, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            fix = self.get(fix_id)
            if fix is None:
                return None
            fix.update(fields)
            self._save(fix)
            return fix

    def mark_applied(self, fix_id: str, job_id: str) -> dict[str, Any] | None:
        """The fix was applied: ``job_id`` is the rerun with it."""
        return self._update(fix_id, applied_job=job_id)

    def set_outcome(self, fix_id: str, outcome: str) -> dict[str, Any] | None:
        """What its rerun showed: "fixed", "not_fixed" or "unknown" (stopped otherwise)."""
        return self._update(fix_id, outcome=outcome)

    def busy(self, job_id: str) -> bool:
        fix = self.latest(job_id)
        return fix is not None and fix["state"] in (WAITING, WORKING)

    def claim(self) -> dict[str, Any] | None:
        """The oldest waiting request, now marked as being worked on."""
        with self._lock:
            fix = next((f for f in self._all() if f["state"] == WAITING), None)
            if fix is not None:
                fix["state"], fix["updated_at"] = WORKING, self.clock()
                self._save(fix)
            return fix

    def _end(self, fix: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        fix["result"] = result
        fix["state"] = DONE if result.get("ok") else FAILED
        fix["updated_at"] = self.clock()
        self._save(fix)
        return fix

    def finish(self, fix_id: str, result: Any) -> dict[str, Any] | None:
        """Record an answer (checked against the request's code); None if there is no such
        request still waiting for one."""
        with self._lock:
            fix = self.get(fix_id)
            if fix is None or fix["state"] not in (WAITING, WORKING):
                return None
            return self._end(fix, checked(FixRequest.from_json(fix["request"]), result))

    # ---------------------------------------------------------------- the helper

    def seen(self, tools: Any, version: str = "") -> None:
        with self._lock:
            self._helper = {"last_seen": self.clock(), "tools": clean_tools(tools),
                            "version": version[:40]}  # fmt: skip

    def helper(self) -> dict[str, Any]:
        with self._lock:
            info = dict(self._helper or {})
        last = info.get("last_seen")
        info["connected"] = last is not None and self.clock() - float(last) <= HELPER_SEEN_S
        return info


def clean_tools(tools: Any) -> list[dict[str, Any]]:
    """The AI tools a helper reports, as kept and shown (known fields, bounded)."""
    return [
        {"name": str(t.get("name", ""))[:40], "label": str(t.get("label", ""))[:40],
         "version": str(t.get("version", ""))[:40], "signed_in": bool(t.get("signed_in")),
         "plan": (str(t["plan"])[:20] if t.get("plan") else None)}
        for t in (tools if isinstance(tools, list) else [])[:MAX_TOOLS]
        if isinstance(t, dict)
    ]  # fmt: skip


def checked(req: FixRequest, result: Any) -> dict[str, Any]:
    """An answer as it is kept: either an error, or an explanation and changes recomputed
    from the request's own code (only each change's cell and new source are taken)."""
    if not isinstance(result, dict):
        return {"ok": False, "error": "the helper sent an answer in an unexpected form"}
    meta: dict[str, Any] = {k: str(result.get(k, ""))[:80] for k in ("tool", "model")}
    seconds = result.get("seconds")
    if isinstance(seconds, int | float):
        meta["seconds"] = round(float(seconds), 1)
    tried = result.get("tried")
    if isinstance(tried, list):
        meta["tried"] = [str(t)[:80] for t in tried[:6]]
    step = result.get("step")
    if isinstance(step, dict):
        model = step.get("model")
        meta["step"] = {"tool": str(step.get("tool", ""))[:40],
                        "model": None if model is None else str(model)[:40]}  # fmt: skip
    if not result.get("ok"):
        return {"ok": False, "error": str(result.get("error") or "the AI gave no fix")[:500],
                **meta}  # fmt: skip
    raw = result.get("changes")
    answer = {
        "explanation": result.get("explanation", ""),
        "changes": [
            {"cell": c.get("cell"), "source": c.get("after", c.get("source", ""))}
            for c in (raw if isinstance(raw, list) else [])
            if isinstance(c, dict)
        ],
    }
    try:
        explanation, changes = changes_from(req, answer)
    except FixError as exc:
        return {"ok": False, "error": str(exc), **meta}
    return {"ok": True, "explanation": explanation, "changes": changes, **meta}


def public(fix: dict[str, Any] | None) -> dict[str, Any] | None:
    """A request as the job page shows it (without the code that was sent)."""
    if fix is None:
        return None
    return {k: fix.get(k) for k in ("id", "job_id", "state", "created_at", "updated_at",
                                    "model", "plan", "auto", "applied_job", "outcome",
                                    "result")}  # fmt: skip
