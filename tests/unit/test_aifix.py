"""AI fix (spec §20, plan A1): the request, Claude Code, the fix board, the job page API,
the website's helper routes and the helper loop. No AI is called: Claude Code is faked."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
import httpx
from fastapi.testclient import TestClient
from test_controller import Harness, _harness
from test_gateway_children import PW, World

from gpurelay.accounts.sessions import SessionSigner
from gpurelay.aifix import choice, claude_code
from gpurelay.aifix.board import DONE, FAILED, WAITING, WORKING, FixBoard, checked
from gpurelay.aifix.helper import HelperRefused, run
from gpurelay.aifix.request import FixError, FixRequest, changes_from, code_cells, prompt
from gpurelay.bundle.packager import upload_bundle
from gpurelay.controller.service import Context
from gpurelay.dashboard.aifix_routes import AiFix
from gpurelay.dashboard.app import ACTION_HEADER, LOCAL_HOSTS, create_app
from gpurelay.gateway.app import GatewaySettings, create_gateway
from gpurelay.models import JobState

ACT = {ACTION_HEADER: "1"}
SCRIPT = "x = 1\ny = x / 0\nprint(y)\n"
ERROR = "line 2: ZeroDivisionError: division by zero"


def _req(**over: Any) -> FixRequest:
    base: dict[str, Any] = dict(
        job_id="job-1",
        name="train",
        entry="train.py",
        cells=[SCRIPT],
        error=ERROR,
        line=2,
        log_tail="Traceback ...\nZeroDivisionError",
    )
    return FixRequest(**{**base, **over})


def _notebook(*sources: str) -> bytes:
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("# notes")]
    nb.cells += [nbformat.v4.new_code_cell(s) for s in sources]
    return nbformat.writes(nb).encode()


# ---------------------------------------------------------------- the request and the answer


def test_code_cells_count_like_the_error_location() -> None:
    assert code_cells(SCRIPT.encode(), "train.py") == [SCRIPT]
    assert code_cells(_notebook("a = 1", "b = a / 0"), "job.ipynb") == ["a = 1", "b = a / 0"]


def test_the_prompt_has_the_error_its_place_the_traceback_and_every_cell() -> None:
    req = _req(
        entry="job.ipynb",
        cells=["import torch", "b = a / 0"],
        cell=2,
        line=1,
        error="cell 2, line 1: NameError: name 'a' is not defined",
        platform="kaggle",
    )
    text = prompt(req)
    assert "Jupyter notebook (job.ipynb)" in text and "(kaggle)" in text
    assert "Error (code cell 2, line 1):" in text and "NameError" in text
    assert "Traceback" in text
    assert "----- Code cell 1 -----\nimport torch" in text and "----- Code cell 2 -----" in text


def test_an_answer_becomes_checked_changes_with_diffs() -> None:
    req = _req()
    explanation, changes = changes_from(req, {
        "explanation": "x / 0 divides by zero.", "changes": [{"cell": 1, "source": "y = 1\n"}],
    })  # fmt: skip
    assert explanation == "x / 0 divides by zero."
    [change] = changes
    assert change["cell"] == 1 and change["before"] == SCRIPT and change["after"] == "y = 1\n"
    assert "-y = x / 0" in change["diff"] and "+y = 1" in change["diff"]
    # A script may be answered as cell 0; an unchanged "change" is dropped.
    _, changes = changes_from(
        req, {"explanation": "ok", "changes": [{"cell": 0, "source": SCRIPT}]}
    )
    assert changes == []
    for bad in ("nope", {"explanation": "x"}, {"explanation": "", "changes": []},
                {"explanation": "x", "changes": [{"cell": 5, "source": "z"}]},
                {"explanation": "x", "changes": [{"cell": "two", "source": "z"}]},
                {"explanation": "x", "changes": [{"cell": 1, "source": "a"},
                                                 {"cell": 1, "source": "b"}]}):  # fmt: skip
        with pytest.raises(FixError):
            changes_from(req, bad)


def test_code_too_long_for_one_prompt_is_not_sent() -> None:
    with pytest.raises(FixError, match="too long"):
        _req(cells=["x" * 400_000]).check_size()


# ---------------------------------------------------------------- Claude Code


class FakeRun:
    """subprocess.run for ``claude``: records calls, answers from a script."""

    def __init__(self, stdout: str = "", stderr: str = "", code: int = 0) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.stdout, self.stderr, self.code = stdout, stderr, code

    def __call__(self, args: list[str], **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, "2.1.312 (Claude Code)\n", "")
        if args[1:3] == ["auth", "status"]:
            out = json.dumps({"loggedIn": True, "subscriptionType": "pro", "email": "a@b.c"})
            return subprocess.CompletedProcess(args, 0, out, "")
        return subprocess.CompletedProcess(args, self.code, self.stdout, self.stderr)


def _which(name: str) -> str | None:
    return "/usr/bin/claude" if name == "claude" else None


def test_claude_code_is_found_with_its_plan_but_not_the_account() -> None:
    info = claude_code.detect(run=FakeRun(), which=_which)
    assert info is not None and info.signed_in and info.plan == "pro"
    assert info.version == "2.1.312" and "a@b.c" not in json.dumps(info.to_json())
    assert claude_code.detect(run=FakeRun(), which=lambda name: None) is None


def test_claude_code_runs_with_no_tools_in_an_empty_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("GPURELAY_VAULT_KEY", "vault")
    answer = {"explanation": "e", "changes": []}
    fake = FakeRun(json.dumps({"is_error": False, "structured_output": answer,
                               "modelUsage": {"claude-sonnet-5-5": {}}}))  # fmt: skip
    got, model = claude_code.ask("the prompt", "sonnet", run=fake, which=_which)
    assert got == answer and model == "claude-sonnet-5-5"
    [(args, kwargs)] = fake.calls
    assert args[args.index("--tools") + 1] == ""  # no tools at all: no files, no commands
    assert "--safe-mode" in args and "--no-session-persistence" in args
    assert json.loads(args[args.index("--json-schema") + 1])["required"] == [
        "explanation",
        "changes",
    ]
    assert kwargs["input"] == "the prompt" and Path(kwargs["cwd"]).name.startswith(
        "gpurelay-aifix-"
    )
    assert "HF_TOKEN" not in kwargs["env"] and "GPURELAY_VAULT_KEY" not in kwargs["env"]


@pytest.mark.parametrize(
    ("stdout", "stderr", "message"),
    [
        (json.dumps({"is_error": True, "result": "Claude usage limit reached"}), "", "usage limit"),
        ("", "Error: not logged in\n", "not logged in"),
        (json.dumps({"is_error": False, "result": "no json here"}), "", "expected form"),
    ],
)
def test_claude_code_failures_say_why(stdout: str, stderr: str, message: str) -> None:
    with pytest.raises(FixError, match=message):
        claude_code.ask("p", "sonnet", run=FakeRun(stdout, stderr, 1), which=_which)


def test_claude_code_answer_in_the_result_text_is_accepted() -> None:
    answer = {"explanation": "e", "changes": []}
    fake = FakeRun(json.dumps({"is_error": False, "result": json.dumps(answer)}))
    assert claude_code.ask("p", "opus", run=fake, which=_which) == (answer, "opus")


def test_a_fix_reports_changes_or_the_reason() -> None:
    good = claude_code.fix(
        _req(),
        asker=lambda text, model: (
            {"explanation": "divides by zero", "changes": [{"cell": 1, "source": "y = 1\n"}]},
            model,
        ),
    )
    assert good["ok"] and good["model"] == "sonnet" and good["changes"][0]["after"] == "y = 1\n"

    def broken(text: str, model: str) -> Any:
        raise FixError("Claude Code: usage limit")

    bad = claude_code.fix(_req(), asker=broken)
    assert bad == {"ok": False, "error": "Claude Code: usage limit", "tool": "claude_code",
                   "model": "sonnet", "seconds": bad["seconds"]}  # fmt: skip


# ---------------------------------------------------------------- the board


def test_board_requests_wait_are_collected_and_answered(tmp_path: Path) -> None:
    now = [100.0]
    board = FixBoard(tmp_path / "aifix", clock=lambda: now[0])
    fix = board.create(_req(), "sonnet")
    assert board.busy("job-1") and board.latest("job-1")["state"] == WAITING  # type: ignore[index]
    claimed = board.claim()
    assert claimed is not None and claimed["id"] == fix["id"] and claimed["state"] == WORKING
    assert board.claim() is None
    # Only the new source of each change is taken; before and diff come from the request.
    done = board.finish(fix["id"], {"ok": True, "explanation": "e", "model": "m", "changes": [
        {"cell": 1, "after": "y = 1\n", "before": "FAKE", "diff": "FAKE"}]})  # fmt: skip
    assert done is not None and done["state"] == DONE
    [change] = done["result"]["changes"]
    assert change["before"] == SCRIPT and "FAKE" not in change["diff"]
    assert board.finish(fix["id"], {"ok": True}) is None  # answered already
    assert FixBoard(tmp_path / "aifix").latest("job-1")["state"] == DONE  # type: ignore[index]


def test_board_gives_up_on_a_request_nobody_answers(tmp_path: Path) -> None:
    now = [0.0]
    board = FixBoard(tmp_path, clock=lambda: now[0])
    board.create(_req(), "sonnet")
    board.claim()
    now[0] += 16 * 60
    fix = board.latest("job-1")
    assert fix is not None and fix["state"] == FAILED and "in time" in fix["result"]["error"]


def test_bad_answers_are_kept_as_failures() -> None:
    assert checked(_req(), "nope")["ok"] is False
    result = checked(
        _req(), {"ok": True, "explanation": "e", "changes": [{"cell": 9, "after": "z"}]}
    )
    assert result["ok"] is False and "does not exist" in result["error"]
    assert checked(_req(), {"ok": False, "error": "x" * 900})["error"] == "x" * 500


def test_the_helper_is_connected_while_it_keeps_asking(tmp_path: Path) -> None:
    now = [0.0]
    board = FixBoard(tmp_path, clock=lambda: now[0])
    assert board.helper()["connected"] is False
    board.seen([{"name": "claude_code", "label": "Claude Code", "version": "2", "signed_in": True,
                 "plan": "pro", "email": "a@b.c"}], "0.1")  # fmt: skip
    info = board.helper()
    assert info["connected"] and "email" not in info["tools"][0]
    now[0] += 31
    assert board.helper()["connected"] is False


# ---------------------------------------------------------------- the job page API


def _failed_job(tmp_path: Path, h: Harness) -> str:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "train.py").write_text(SCRIPT, encoding="utf-8")
    job = h.submit("train")
    upload_bundle(h.storage, job.id, project / "train.py", include_wheel=False)
    h.db.put_job(job.model_copy(update={"state": JobState.FAILED_USER, "message": ERROR}))
    return job.id


def _fixed(req: FixRequest, model: str) -> dict[str, Any]:
    return claude_code.fix(req, model, asker=lambda text, m: (
        {"explanation": "x / 0 divides by zero", "changes": [{"cell": 1, "source": "y = 1\n"}]},
        "claude-sonnet-5-5"))  # fmt: skip


def _solver(req: FixRequest, steps: list[dict[str, Any]], available: Any = None) -> dict[str, Any]:
    return choice.solve(req, steps, available, fixers={"claude_code": _fixed})


INFO = claude_code.ToolInfo("claude_code", "Claude Code", "2", True, "pro")


def _ai(tmp_path: Path, h: Harness, mode: str, **options: Any) -> tuple[AiFix, Context]:
    ctx = Context(h.controller.config, tmp_path / "home", h.db, h.storage, h.controller)

    def find(job_id: str) -> Any:
        from fastapi import HTTPException

        job = h.db.get_job(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    options.setdefault("solver", _solver)
    options.setdefault("detect", lambda: [INFO])
    return AiFix(ctx, find, mode, action_header=ACTION_HEADER, **options), ctx


def _app(tmp_path: Path, h: Harness, mode: str, **options: Any) -> TestClient:
    ai, ctx = _ai(tmp_path, h, mode, **options)
    app = create_app(ctx, clock=h.clock, allowed_hosts=LOCAL_HOSTS | {"testserver"}, ai=ai)
    return TestClient(app)


def _wait_done(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(200):
        fix = client.get(f"/api/jobs/{job_id}/fix").json()["fix"]
        if fix["state"] in (DONE, FAILED):
            return dict(fix)
        time.sleep(0.02)
    raise AssertionError("the fix never finished")


def test_fix_it_with_ai_on_this_computer(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    job_id = _failed_job(tmp_path, h)
    client = _app(tmp_path, h, "local")
    ai = client.get("/api/ai").json()
    assert ai["mode"] == "local" and ai["ready"] is True and ai["tools"] == [INFO.to_json()]
    assert ai["settings"] == {"auto": False, "max_attempts": 2, "max_gpu_hours": 6.0}
    assert client.get(f"/api/jobs/{job_id}/fix").json()["fix"] is None
    assert client.post(f"/api/jobs/{job_id}/fix").status_code == 403  # the action header
    started = client.post(f"/api/jobs/{job_id}/fix", headers=ACT).json()["fix"]
    assert started["state"] in (WAITING, WORKING) and "request" not in started  # code not echoed
    fix = _wait_done(client, job_id)
    assert fix["state"] == DONE and fix["result"]["model"] == "claude-sonnet-5-5"
    assert fix["result"]["explanation"] == "x / 0 divides by zero"
    assert "+y = 1" in fix["result"]["changes"][0]["diff"]


def test_only_a_job_that_failed_in_its_code_can_be_fixed(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    running = h.submit("other")
    client = _app(tmp_path, h, "local", detect=lambda: [])
    assert client.post(f"/api/jobs/{running.id}/fix", headers=ACT).status_code == 409
    assert client.get("/api/ai").json()["ready"] is False
    assert client.post("/api/helper/poll", headers=ACT, json={}).status_code in (404, 405)


def test_a_job_whose_code_is_gone_says_so(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    job = h.submit("lost")
    h.db.put_job(job.model_copy(update={"state": JobState.FAILED_USER, "message": ERROR}))
    response = _app(tmp_path, h, "local").post(f"/api/jobs/{job.id}/fix",
                                                             headers=ACT)  # fmt: skip
    assert response.status_code == 422 and "could not be read" in response.json()["detail"]


def test_the_page_offers_both_choices() -> None:
    from importlib import resources

    page = resources.files("gpurelay.dashboard").joinpath("index.html").read_text("utf-8")
    assert "Fix it with AI" in page and "Check it myself" in page
    assert "stopped with an error in its code" in page  # the browser notification


# ---------------------------------------------------------------- the website and the helper


@pytest.fixture
def site(tmp_path: Path) -> Any:
    """The website with each user's GPU Relay being the real web app (helper mode)."""
    w = World(tmp_path)
    h = _harness(tmp_path / "jobs", {"kaggle": (30.0, True, 1, "T4")})
    ctx = Context(h.controller.config, tmp_path / "home", h.db, h.storage, h.controller)
    user_app = create_app(ctx, clock=h.clock, allowed_hosts=LOCAL_HOSTS, ai_fix="helper")
    transport = httpx.ASGITransport(app=user_app)
    settings = GatewaySettings(
        store=w.store, signer=SessionSigner(b"k" * 32), spaces=w.spaces, children=w.children,
        secure_cookie=False,
        proxy_client=lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout),
    )  # fmt: skip
    return w, h, TestClient(create_gateway(settings), follow_redirects=False)


class FakeSite:
    """The helper's client, talking to the test website (as SiteClient does over HTTPS)."""

    def __init__(self, client: TestClient, key: str) -> None:
        self.client, self.key, self.site = client, key, "https://test"

    def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(path, json=data,
                                    headers={"Authorization": f"Bearer {self.key}"})  # fmt: skip
        if response.status_code == 401:
            raise HelperRefused(response.json()["detail"])
        if response.status_code >= 400:
            raise ConnectionError(response.text)
        return dict(response.json())


def test_from_the_job_page_through_the_helper_and_back(site: Any, tmp_path: Path) -> None:
    w, h, client = site
    w.user("ada@example.com")
    client.post("/signin", data={"email": "ada@example.com", "password": PW})
    assert client.get("/api/ai/helper").json()["key"] is False
    key = client.post("/api/ai/helper/key").json()["key"]
    assert key.startswith("grh_") and client.get("/api/ai/helper").json()["key"] is True
    assert key not in json.dumps(w.store._all("SELECT * FROM links"))  # only its hash is kept

    job_id = _failed_job(tmp_path, h)
    asked = client.post(f"/app/api/jobs/{job_id}/fix", headers=ACT)
    assert asked.status_code == 200 and asked.json()["fix"]["state"] == WAITING

    said: list[str] = []
    code = run(FakeSite(client, key), once=True, detect=lambda: [INFO], solver=_solver,
               say=said.append)  # fmt: skip
    assert code == 0 and any("proposed 1 change" in s for s in said), said

    fix = client.get(f"/app/api/jobs/{job_id}/fix").json()["fix"]
    assert fix["state"] == DONE and "+y = 1" in fix["result"]["changes"][0]["diff"]
    status = client.get("/api/ai/helper").json()
    assert status["connected"] and status["tools"][0]["plan"] == "pro"
    ai = client.get("/app/api/ai").json()
    assert ai["mode"] == "helper" and ai["ready"] is True


def test_the_helper_needs_a_valid_key_and_never_starts_a_stopped_app(site: Any) -> None:
    w, _, client = site
    ada = w.user("ada@example.com")
    assert client.post("/helper/v1/poll", json={}).status_code == 401
    bad = {"Authorization": "Bearer grh_wrong"}
    assert client.post("/helper/v1/poll", json={}, headers=bad).status_code == 401
    client.post("/signin", data={"email": "ada@example.com", "password": PW})
    key = client.post("/api/ai/helper/key").json()["key"]
    client.cookies.clear()  # the helper has no session, only its key
    good = {"Authorization": f"Bearer {key}"}
    assert client.post("/helper/v1/poll", json={"tools": []}, headers=good).json() == {
        "fix": None, "app": "stopped"}  # fmt: skip
    assert w.children.get(ada) is None  # polling did not start their GPU Relay
    client.post("/signin", data={"email": "ada@example.com", "password": PW})
    client.post("/api/ai/helper/remove")
    assert client.post("/helper/v1/poll", json={}, headers=good).status_code == 401


def test_wrong_helper_keys_are_rate_limited(site: Any) -> None:
    _, _, client = site
    bad = {"Authorization": "Bearer grh_wrong"}
    codes = [client.post("/helper/v1/poll", json={}, headers=bad).status_code for _ in range(12)]
    assert codes[:10] == [401] * 10 and codes[-1] == 429


def test_the_helper_stops_on_a_refused_key_and_reports_a_missing_tool() -> None:
    class Script:
        site = "https://test"

        def __init__(self, answers: list[Any]) -> None:
            self.answers, self.sent = answers, []

        def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
            self.sent.append((path, data))
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return dict(answer)

    refused = Script([HelperRefused("unknown helper key")])
    assert run(refused, detect=lambda: [], say=lambda s: None, sleep=lambda s: None) == 2  # type: ignore[arg-type]

    fix = {"id": "fix-1", "model": "sonnet", "request": _req().to_json()}
    no_tool = Script([{"fix": fix}, {"ok": True}, {"fix": None}])
    assert run(no_tool, once=True, detect=lambda: [], say=lambda s: None) == 0  # type: ignore[arg-type]
    path, data = no_tool.sent[1]
    assert path == "/helper/v1/fixes/fix-1" and data["ok"] is False
    assert "install Claude Code" in data["error"]

    offline = Script([OSError("no network")])
    assert run(offline, once=True, detect=lambda: [], say=lambda s: None) == 1  # type: ignore[arg-type]


def test_the_setup_page_has_the_ai_step() -> None:
    from importlib import resources

    page = resources.files("gpurelay.gateway").joinpath("setup.html").read_text("utf-8")
    assert 'id="ai"' in page and "gpurelay helper --site" in page and "/api/ai/helper/key" in page


def test_the_helpers_http_client_sends_its_key_and_reads_refusals() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from gpurelay.aifix.helper import SiteClient

    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append({"path": self.path, "auth": self.headers["Authorization"],
                         "body": json.loads(body)})  # fmt: skip
            ok = self.headers["Authorization"] == "Bearer grh_good"
            answer = {"fix": None} if ok else {"detail": "unknown helper key"}
            data = json.dumps(answer).encode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    site = f"http://127.0.0.1:{server.server_port}"
    try:
        assert SiteClient(site, "grh_good").post("/helper/v1/poll", {"tools": []}) == {"fix": None}
        with pytest.raises(HelperRefused, match="unknown helper key"):
            SiteClient(site, "grh_bad").post("/helper/v1/poll", {})
    finally:
        server.shutdown()
    assert seen[0] == {"path": "/helper/v1/poll", "auth": "Bearer grh_good", "body": {"tools": []}}


def test_the_helper_command_needs_a_site_and_a_key(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from gpurelay.cli import app

    env = {"GPURELAY_HOME": str(tmp_path), "GPURELAY_HELPER_KEY": ""}
    result = CliRunner().invoke(app, ["helper"], env=env)
    assert result.exit_code == 1 and "--site" in result.output
    result = CliRunner().invoke(app, ["helper", "--site", "http://example.com"], env=env)
    assert result.exit_code == 1 and "https://" in result.output
