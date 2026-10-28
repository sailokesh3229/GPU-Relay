"""AI fix, plans A2 (apply and rerun, automatic reruns) and A3 (Codex, model choice,
escalation, the record of fixes). No AI is called."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from test_aifix import (
    ACT,
    ERROR,
    INFO,
    SCRIPT,
    FakeRun,
    _ai,
    _app,
    _failed_job,
    _notebook,
    _req,
    _wait_done,
    _which,
)
from test_aifix import site as site  # a fixture
from test_controller import _harness
from test_gateway_children import PW

from gpurelay.aifix import choice, codex
from gpurelay.aifix.apply import chain, fixed_entry
from gpurelay.aifix.board import DONE
from gpurelay.aifix.request import FixError
from gpurelay.bundle.packager import read_bundle_files, upload_bundle
from gpurelay.config import AiFixSettings, load_config
from gpurelay.dashboard import aifix_routes
from gpurelay.dashboard.aifix_routes import outcome
from gpurelay.models import JobState

CLAUDE_ONLY = [INFO.to_json()]
BOTH = [INFO.to_json(), {"name": "codex", "label": "Codex", "version": "0.1", "signed_in": True}]


# ---------------------------------------------------------------- A2: applying a fix


def test_a_fix_goes_into_the_notebook_with_old_outputs_cleared() -> None:
    import nbformat

    source = _notebook("a = 1", "b = a / 0")
    nb = nbformat.reads(source.decode(), 4)
    nb.cells[2].outputs = [nbformat.v4.new_output("stream", text="old")]
    fixed = fixed_entry("job.ipynb", nbformat.writes(nb).encode(),
                        [{"cell": 2, "after": "b = a / 1"}])  # fmt: skip
    cells = nbformat.reads(fixed.decode(), 4).cells
    assert cells[0].cell_type == "markdown" and cells[2].source == "b = a / 1"
    assert cells[2].outputs == [] and cells[1].source == "a = 1"
    assert fixed_entry("train.py", b"x", [{"cell": 1, "after": "y"}]) == b"y"
    with pytest.raises(FixError):
        fixed_entry("job.ipynb", source, [{"cell": 7, "after": "z"}])


def _with_extras(tmp_path: Path, h: Any) -> str:
    """A failed job whose bundle has requirements and a second module."""
    project = tmp_path / "proj2"
    project.mkdir()
    (project / "train.py").write_text("import helpers\n" + SCRIPT, encoding="utf-8")
    (project / "helpers.py").write_bytes(b"VALUE = 2\n")
    (project / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    job = h.submit("train", priority=3, prefer_backend="kaggle", max_paid_usd=1.5)
    upload_bundle(h.storage, job.id, project / "train.py", {"helpers.py": project / "helpers.py"},
                  requirements=project / "requirements.txt", include_wheel=False)  # fmt: skip
    h.db.put_job(job.model_copy(update={"state": JobState.FAILED_USER,
                                        "message": "line 3: ZeroDivisionError: x"}))  # fmt: skip
    return job.id


@pytest.fixture
def no_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reruns are uploaded without the runtime wheel (never run here; building it is slow)."""
    from gpurelay.bundle import packager

    real = packager.upload_bundle

    def upload(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **{**kwargs, "include_wheel": False})

    monkeypatch.setattr("gpurelay.controller.service.upload_bundle", upload)


@pytest.mark.usefixtures("no_wheel")
def test_apply_and_rerun_submits_a_fixed_copy_as_a_new_attempt(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    job_id = _with_extras(tmp_path, h)
    client = _app(tmp_path, h, "local")
    client.post(f"/api/jobs/{job_id}/fix", headers=ACT)
    fix = _wait_done(client, job_id)
    assert fix["state"] == DONE
    url = f"/api/jobs/{job_id}/fix/{fix['id']}/apply"
    assert client.post(url).status_code == 403  # the action header
    applied = client.post(url, headers=ACT).json()
    new = h.db.get_job(applied["job_id"])
    assert new is not None and applied["name"] == "train · fix 1"
    assert (new.retry_of, new.fix_id) == (job_id, fix["id"])
    assert (new.priority, new.prefer_backend, new.max_paid_usd) == (3, "kaggle", 1.5)
    manifest, files = read_bundle_files(h.storage, new.id)
    assert files["train.py"] == b"y = 1\n"  # the fix
    assert files["helpers.py"] == b"VALUE = 2\n" and manifest.requirements == "requirements.txt"
    assert h.db.get_job(job_id).state is JobState.FAILED_USER  # type: ignore[union-attr]
    again = client.post(url, headers=ACT)
    assert again.status_code == 409 and "applied already" in again.json()["detail"]
    page = client.get(f"/api/jobs/{job_id}/fix").json()
    assert page["fix"]["applied_job"] == new.id
    assert page["attempt"]["rerun"]["id"] == new.id
    assert client.get(f"/api/jobs/{new.id}").json()["job"]["retry_of"] == job_id
    assert client.get(f"/api/jobs/{new.id}/fix").json()["attempt"]["number"] == 2


# ---------------------------------------------------------------- A2: automatic reruns


def _finish(h: Any, job_id: str, state: JobState, message: str = "") -> None:
    job = h.db.get_job(job_id)
    h.db.put_job(job.model_copy(update={"state": state, "message": message}))


@pytest.mark.usefixtures("no_wheel")
def test_automatic_reruns_within_the_limits_and_escalation(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    first = _failed_job(tmp_path, h)
    asked: list[list[dict[str, Any]]] = []

    def solver(req: Any, steps: list[dict[str, Any]], available: Any = None) -> dict[str, Any]:
        asked.append(steps)
        new = f"y = {len(asked)}\n"  # a different change each time

        def fixer(req: Any, model: str | None) -> dict[str, Any]:
            from gpurelay.aifix import claude_code

            return claude_code.fix(req, model, asker=lambda text, m: (
                {"explanation": "e", "changes": [{"cell": 1, "source": new}]}, str(m)))  # fmt: skip

        return choice.solve(req, steps, available, fixers={"claude_code": fixer})

    ai, ctx = _ai(tmp_path, h, "local", solver=solver)
    ai.tick()  # auto is off: nothing happens
    assert ai.board.latest(first) is None
    ctx.config.aifix = AiFixSettings(auto=True, max_attempts=2)

    ai.tick()  # a fix is asked for
    fix = ai.board.latest(first)
    assert fix is not None and fix["auto"] and ai.busy()
    _wait_board(ai, first)
    ai.tick()  # ... and applied
    [rerun] = [j for j in h.db.list_jobs() if j.retry_of == first]
    _finish(h, rerun.id, JobState.FAILED_USER, ERROR)  # the same error again
    ai.tick()  # outcome recorded; the rerun gets the next (escalated) fix
    assert ai.board.latest(first)["outcome"] == "not_fixed"  # type: ignore[index]
    assert ai.stats.load()["ZeroDivisionError"]["claude_code/opus"] == {"tried": 1, "fixed": 0}
    _wait_board(ai, rerun.id)
    assert asked[0][0]["model"] == "opus" and asked[1][0]["model"] == "sonnet"  # escalated
    ai.tick()
    [third] = [j for j in h.db.list_jobs() if j.retry_of == rerun.id]
    assert third.name == "train · fix 2" and len(chain(ctx, third)) == 3
    _finish(h, third.id, JobState.FAILED_USER, ERROR)
    ai.tick()  # two fixes made: the limit
    assert ai.board.latest(third.id) is None and not ai.busy()


def _wait_board(ai: Any, job_id: str) -> None:
    import time

    for _ in range(300):
        fix = ai.board.latest(job_id)
        if fix is not None and fix["state"] in ("done", "failed"):
            return
        time.sleep(0.01)
    raise AssertionError("no answer")


def test_automatic_reruns_stop_at_the_gpu_hours_limit(tmp_path: Path, monkeypatch: Any) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    job_id = _failed_job(tmp_path, h)
    ai, ctx = _ai(tmp_path, h, "local")
    ctx.config.aifix = AiFixSettings(auto=True, max_gpu_hours=1.0)
    monkeypatch.setattr(aifix_routes, "gpu_hours", lambda ctx, jobs, now: 1.5)
    ai.tick()
    assert ai.board.latest(job_id) is None


def test_old_failures_are_not_rerun_automatically(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    job_id = _failed_job(tmp_path, h)
    ai, ctx = _ai(tmp_path, h, "local", wall=lambda: 4e9)  # far in the future
    ctx.config.aifix = AiFixSettings(auto=True)
    ai.tick()
    assert ai.board.latest(job_id) is None


def test_settings_change_here_and_are_saved(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    client = _app(tmp_path, h, "local")
    body = {"auto": True, "max_attempts": 3, "max_gpu_hours": 4}
    assert client.post("/api/ai/settings", json=body).status_code == 403
    assert client.post("/api/ai/settings", json=body, headers=ACT).json() == {"ok": True}
    saved = load_config(tmp_path / "home" / "config.toml").aifix
    assert (saved.auto, saved.max_attempts, saved.max_gpu_hours) == (True, 3, 4.0)
    assert client.get("/api/ai").json()["settings"]["auto"] is True
    bad = client.post("/api/ai/settings", json={**body, "max_attempts": 9}, headers=ACT)
    assert bad.status_code == 422
    helper = _app(tmp_path / "w", _harness(tmp_path / "w", {"k": (1.0, True, 1, "T4")}), "helper")
    assert helper.post("/api/ai/settings", json=body, headers=ACT).status_code == 409


def test_the_website_keeps_a_busy_users_gpu_relay_running(tmp_path: Path) -> None:
    from test_gateway_children import World

    w = World(tmp_path)
    ada = w.user("ada@example.com")
    child = w.children.ensure(ada)
    w.now += 10_000  # long idle
    w.children.get_json = lambda url, key: {"jobs": [], "ai_busy": True}  # type: ignore[method-assign]
    w.children.tick()
    assert w.children.get(ada) is child and child.active
    w.children.get_json = lambda url, key: {"jobs": []}  # type: ignore[method-assign]
    w.children.tick()
    assert w.children.get(ada) is None


def test_on_the_website_auto_settings_are_in_setup_and_reruns_count_as_jobs(site: Any) -> None:
    w, _, client = site
    ada = w.user("ada@example.com")
    client.post("/signin", data={"email": "ada@example.com", "password": PW})
    saved = client.post("/api/setup/aifix", json={"auto": True, "max_attempts": 3,
                                                  "max_gpu_hours": 2})  # fmt: skip
    assert saved.status_code == 200
    assert client.get("/api/setup").json()["steps"]["aifix"] == {
        "auto": True, "max_attempts": 3, "max_gpu_hours": 2.0}  # fmt: skip
    assert w.spaces.config(ada).aifix.auto is True
    w.children.ensure(ada)
    w.children.jobs_in_flight = lambda user_id: 3  # type: ignore[method-assign]
    limited = client.post("/app/api/jobs/job-1/fix/fix-1/apply", headers=ACT)
    assert limited.status_code == 429


# ---------------------------------------------------------------- A3: which model


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ("cell 2, line 1: NameError: name 'a' is not defined", "NameError"),
        ("line 3: RuntimeError: mat1 and mat2 shapes cannot be multiplied", "RuntimeError"),
        (
            "cell 4, line 9: OutOfMemoryError: CUDA out of memory. Tried to allocate",
            "OutOfMemoryError",
        ),
        ("RuntimeError: CUDA out of memory", "OutOfMemoryError"),
        ("final checkpoint failed", "Error"),
    ],
)
def test_the_kind_of_error(error: str, kind: str) -> None:
    assert choice.error_kind(error) == kind


def test_simple_errors_go_to_the_fast_model_hard_ones_to_the_strongest() -> None:
    s = AiFixSettings()
    simple = _req(error="line 2: NameError: name 'score' is not defined")
    hard = _req(error="line 2: RuntimeError: shapes cannot be multiplied")
    assert choice.plan(simple, s, CLAUDE_ONLY) == [
        {"tool": "claude_code", "model": "sonnet"},
        {"tool": "claude_code", "model": "opus"},
    ]
    assert [st["model"] for st in choice.plan(hard, s, BOTH)] == ["opus", None, "sonnet"]
    assert [st["tool"] for st in choice.plan(hard, s, BOTH)] == ["claude_code", "codex",
                                                                 "claude_code"]  # fmt: skip
    # Before a helper reports its tools, Claude Code is assumed.
    assert choice.plan(hard, s, [])[0] == {"tool": "claude_code", "model": "opus"}
    # Escalation: one earlier fix that did not work moves the plan on.
    assert choice.plan(simple, s, CLAUDE_ONLY, earlier=1)[0]["model"] == "opus"
    assert choice.plan(simple, s, BOTH, earlier=9)[0]["tool"] == "codex"  # the last step


def test_a_model_that_fixed_this_kind_more_often_goes_first(tmp_path: Path) -> None:
    stats = choice.Stats(tmp_path / "stats.json")
    hard = _req(error="line 2: RuntimeError: shapes")
    for fixed in (True, True, True):
        stats.record("RuntimeError", "codex", "default", fixed)
    for fixed in (True, False, False):
        stats.record("RuntimeError", "claude_code", "opus", fixed)
    assert stats.rate("RuntimeError", "codex", "default") == 1.0
    assert choice.plan(hard, AiFixSettings(), BOTH, stats)[0]["tool"] == "codex"
    assert choice.plan(hard, AiFixSettings(), CLAUDE_ONLY, stats)[0]["model"] == "opus"
    stats.record("NameError", "codex", "default", True)  # too few tries to count
    assert stats.rate("NameError", "codex", "default") is None


def test_solve_falls_back_to_the_next_step() -> None:
    calls: list[str] = []

    def limited(req: Any, model: str | None) -> dict[str, Any]:
        calls.append(f"claude/{model}")
        return {"ok": False, "error": "Claude Code: usage limit reached", "model": model}

    def works(req: Any, model: str | None) -> dict[str, Any]:
        calls.append("codex")
        return {"ok": True, "tool": "codex", "model": "default", "explanation": "e",
                "changes": []}  # fmt: skip

    steps = [{"tool": "claude_code", "model": "opus"}, {"tool": "codex", "model": None}]
    fixers = {"claude_code": limited, "codex": works}
    result = choice.solve(_req(), steps, {"claude_code", "codex"}, fixers)
    assert result["ok"] and result["step"] == {"tool": "codex", "model": None}
    assert result["tried"] == ["claude_code/opus", "codex/default"]
    only_claude = choice.solve(_req(), steps, {"claude_code"}, fixers)
    assert not only_claude["ok"] and "usage limit" in only_claude["error"]
    nothing = choice.solve(_req(), steps, set(), fixers)
    assert not nothing["ok"] and "install Claude Code" in nothing["error"]


@pytest.mark.parametrize(
    ("state", "message", "expected"),
    [
        (JobState.RUNNING, "", None),
        (JobState.COMPLETED, "", "fixed"),
        (JobState.FAILED_USER, ERROR, "not_fixed"),
        (JobState.FAILED_USER, "line 9: KeyError: 'x'", "fixed"),  # got past that error
        (JobState.FAILED_INFRA, "boom", "unknown"),
    ],
)
def test_what_a_rerun_shows(state: JobState, message: str, expected: str | None) -> None:
    from gpurelay.models import Job, JobKind

    rerun = Job(name="r", kind=JobKind.SCRIPT, entry="r.py", state=state, message=message)
    assert outcome(ERROR, rerun) == expected


# ---------------------------------------------------------------- A3: Codex


class CodexRun(FakeRun):
    def __call__(self, args: list[str], **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, "codex-cli 0.160.0\n", "")
        if args[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(args, self.code, "Logged in using ChatGPT\n", "")
        out = Path(args[args.index("-o") + 1])
        if self.stdout:
            out.write_text(self.stdout, encoding="utf-8")
        return subprocess.CompletedProcess(args, self.code, "", self.stderr)


def _codex_which(name: str) -> str | None:
    return "/usr/bin/codex" if name == "codex" else None


def test_codex_is_found_with_its_sign_in() -> None:
    info = codex.detect(run=CodexRun(), which=_codex_which)
    assert info is not None and info.signed_in and info.plan == "chatgpt"
    assert info.version == "0.160.0"
    assert codex.detect(run=CodexRun(code=1), which=_codex_which).signed_in is False  # type: ignore[union-attr]
    assert codex.detect(run=CodexRun(), which=_which) is None


def test_codex_runs_read_only_with_a_schema_and_no_saved_session() -> None:
    answer = {"explanation": "e", "changes": [{"cell": 1, "source": "y = 1\n"}]}
    fake = CodexRun(json.dumps(answer))
    got, model = codex.ask("the prompt", None, run=fake, which=_codex_which)
    assert got == answer and model == codex.DEFAULT_LABEL
    [(args, kwargs)] = fake.calls
    assert args[:3] == ["/usr/bin/codex", "exec", "-"] and kwargs["input"] == "the prompt"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in args and "--skip-git-repo-check" in args and "-m" not in args
    assert Path(kwargs["cwd"]).name == "work"  # an empty folder
    schema = args[args.index("--output-schema") + 1]
    assert schema.endswith("schema.json")
    _, model = codex.ask("p", "gpt-6.1-sol", run=CodexRun(json.dumps(answer)), which=_codex_which)
    assert model == "gpt-6.1-sol"
    with pytest.raises(FixError, match="Codex stopped: not signed in"):
        codex.ask("p", None, run=CodexRun("", "error\nnot signed in\n", 1), which=_codex_which)
    result = codex.fix(_req(), asker=lambda text, m: (answer, "codex-default"))
    assert result["ok"] and result["tool"] == "codex" and "+y = 1" in result["changes"][0]["diff"]
