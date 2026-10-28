"""Dashboard API (plan Phase 8) over the controller test harness: no network, no server."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpurelay.constants import COLAB_LAUNCHER_URL

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from test_controller import Harness, _harness

from gpurelay.controller.service import Context
from gpurelay.dashboard.app import ACTION_HEADER, LOCAL_HOSTS, create_app
from gpurelay.models import JobState
from gpurelay.storage.layout import final_prefix

ACT = {ACTION_HEADER: "1"}


def _client(tmp_path: Path, h: Harness) -> TestClient:
    ctx = Context(h.controller.config, tmp_path / "home", h.db, h.storage, h.controller)
    return TestClient(create_app(ctx, clock=h.clock, allowed_hosts=LOCAL_HOSTS | {"testserver"}))


def test_page_is_self_contained(tmp_path: Path) -> None:
    client = _client(tmp_path, _harness(tmp_path, {"a": (None, True, 1, "T4")}))
    page = client.get("/")
    assert page.status_code == 200 and "<title>GPU Relay</title>" in page.text
    assert "https://" not in page.text and "<script src" not in page.text  # works offline
    assert "innerHTML" not in page.text  # data is inserted as text only


def test_overview_jobs_budgets_and_colab_banner(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4"), "colab": (None, False, 1, "T4")})
    running = h.submit("train")
    h.controller.tick()
    data = _client(tmp_path, h).get("/api/overview").json()
    assert data["controller_running"] is False
    [job] = data["jobs"]
    assert job["id"] == running.id and job["state"] == "running" and job["backend"] == "kaggle"
    budgets = {b["name"]: b for b in data["backends"]}
    assert budgets["kaggle"]["budget_h"] == 30.0 and budgets["kaggle"]["running"] == 1
    assert budgets["colab"]["automated"] is False and budgets["colab"]["remaining"] == "unknown"
    assert data["colab_actions"] == []

    h2 = _harness(tmp_path / "manual", {"colab": (None, False, 1, "T4")})
    waiting = h2.submit("needs-colab")
    h2.controller.tick()
    data = _client(tmp_path / "manual", h2).get("/api/overview").json()
    assert data["colab_actions"] == [
        {
            "job_id": waiting.id,
            "name": "needs-colab",
            "command": f"gpurelay colab {waiting.id}",
            "open_url": COLAB_LAUNCHER_URL,
        }  # one click opens the launcher in Colab
    ]


def test_job_detail_segments_events_and_404(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    h.finish(job.id, 85, "handoff")
    h.controller.tick()  # handoff -> relaunched
    client = _client(tmp_path, h)
    data = client.get(f"/api/jobs/{job.id}").json()
    assert [s["outcome"] for s in data["segments"]] == ["handoff", None]
    assert data["segments"][1]["state"] == "running"
    assert any(e["kind"] == "segment_handoff" for e in data["events"])
    assert data["fit"]
    assert client.get("/api/jobs/job-nope").status_code == 404


def test_log_tail(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    client = _client(tmp_path, h)
    assert client.get(f"/api/jobs/{job.id}/log").json() == {
        "text": "", "length": 0, "final": False,
    }  # fmt: skip
    h.storage.put_bytes(b"line 1\nline 2\nline 3\n", f"{final_prefix(job.id)}/log.txt")
    data = client.get(f"/api/jobs/{job.id}/log", params={"tail": 7}).json()
    assert data == {"text": "line 3\n", "length": 21, "final": False}


def test_actions_need_the_header_and_follow_job_rules(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    h.controller.tick()
    client = _client(tmp_path, h)
    assert client.post(f"/api/jobs/{job.id}/pause").status_code == 403  # e.g. another site
    assert client.post(f"/api/jobs/{job.id}/explode", headers=ACT).status_code == 404
    assert client.post(f"/api/jobs/{job.id}/resume", headers=ACT).status_code == 409
    response = client.post(f"/api/jobs/{job.id}/pause", headers=ACT)
    assert response.status_code == 200 and h.job(job.id).requested == "pause"
    h.db.put_job(h.job(job.id).model_copy(update={"state": JobState.COMPLETED, "requested": None}))
    assert client.post(f"/api/jobs/{job.id}/cancel", headers=ACT).status_code == 409


def test_colab_notebook_download_needs_the_colab_backend(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    job = h.submit()
    response = _client(tmp_path, h).get(f"/api/jobs/{job.id}/colab-notebook")
    assert response.status_code == 409 and "Colab backend is off" in response.json()["detail"]
    assert _client(tmp_path, h).get("/api/jobs/job-nope/colab-notebook").status_code == 404


def test_dns_rebinding_is_refused(tmp_path: Path) -> None:
    h = _harness(tmp_path, {"a": (None, True, 1, "T4")})
    ctx = Context(h.controller.config, tmp_path / "home", h.db, h.storage, h.controller)
    client = TestClient(create_app(ctx, clock=h.clock))  # default: local names only
    assert client.get("/api/overview", headers={"host": "evil.example:8765"}).status_code == 421
    assert client.get("/", headers={"host": "attacker.test"}).status_code == 421
    for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765", "localhost"):
        assert client.get("/api/overview", headers={"host": host}).status_code == 200, host


def test_notebook_results_show_text_figures_and_errors(tmp_path: Path) -> None:
    import nbformat

    from gpurelay.dashboard.app import notebook_results

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# title"),
        nbformat.v4.new_code_cell("print('loss')", outputs=[
            nbformat.v4.new_output("stream", name="stdout", text="Epoch 1\nloss 0.5\n")]),
        nbformat.v4.new_code_cell("plt.show()", outputs=[
            nbformat.v4.new_output("display_data", data={"image/png": "iVBORw0K\nGgo=",
                                                         "text/plain": "<Figure>"})]),
        nbformat.v4.new_code_cell("1/0", outputs=[
            nbformat.v4.new_output("error", ename="ZeroDivisionError", evalue="division by zero",
                                   traceback=["Traceback", "ZeroDivisionError: x"])]),
        nbformat.v4.new_code_cell("x = 1"),
    ]  # fmt: skip
    path = tmp_path / "output.ipynb"
    nbformat.write(nb, str(path))
    cells = notebook_results(path)
    assert [c["source"] for c in cells] == ["print('loss')", "plt.show()", "1/0", "x = 1"]
    assert cells[0]["outputs"] == [{"kind": "text", "text": "Epoch 1\nloss 0.5\n"}]
    assert cells[1]["outputs"] == [{"kind": "image", "png": "iVBORw0KGgo="}]
    assert (
        cells[2]["outputs"][0]["kind"] == "error"
        and "ZeroDivisionError" in cells[2]["outputs"][0]["text"]
    )
    assert cells[3]["outputs"] == []


def test_error_snippet_marks_the_failing_line() -> None:
    import nbformat

    from gpurelay.dashboard.app import error_snippet
    from gpurelay.runtime.errorloc import located, parse_location

    script = "\n".join(f"line{n}" for n in range(1, 21)).encode()
    snip = error_snippet(script, "train.py", None, 10)
    assert snip == {"cell": None, "line": 10, "first": 4,
                    "lines": [f"line{n}" for n in range(4, 17)]}  # fmt: skip
    assert error_snippet(script, "train.py", None, 99) is None
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell("a = 1"), nbformat.v4.new_markdown_cell("m"),
                nbformat.v4.new_code_cell("x = 1\ny = x / 0")]  # fmt: skip
    snip = error_snippet(nbformat.writes(nb).encode(), "job.ipynb", 2, 2)
    assert snip is not None and snip["lines"] == ["x = 1", "y = x / 0"] and snip["first"] == 1
    assert error_snippet(nbformat.writes(nb).encode(), "job.ipynb", 3, 1) is None
    assert parse_location(located("NameError: x", 5, 2)) == (2, 5)
    assert parse_location(located("ValueError: y", 7)) == (None, 7)
    assert parse_location("final checkpoint failed") is None
    assert located("plain", None) == "plain"


def test_the_colab_launcher_notebook_is_valid_and_finds_waiting_jobs() -> None:
    import ast
    import json
    from pathlib import Path

    from IPython.core.inputtransformer2 import TransformerManager

    from gpurelay.constants import COLAB_LAUNCHER_URL

    path = Path(__file__).parents[2] / "colab" / "open_job.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(TransformerManager().transform_cell(source))
    joined = "\n".join(code)
    assert "GPURELAY_HF_TOKEN" in joined and "manual_session.json" in joined
    assert "colab/resume.ipynb" in joined  # what ColabManualBackend uploads (layout.py)
    assert COLAB_LAUNCHER_URL.endswith("/blob/main/colab/open_job.ipynb")


def test_notebook_shell_commands_use_the_jobs_own_python(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import sys
    from pathlib import Path

    from gpurelay.runtime.executor_notebook import job_python_first_on_path

    here = str(Path(sys.executable).parent)
    monkeypatch.setenv("PATH", os.pathsep.join(["/elsewhere/bin", here, "/usr/bin"]))
    job_python_first_on_path()  # `!pip` must not reach another Python installation first
    assert os.environ["PATH"].split(os.pathsep) == [here, "/elsewhere/bin", "/usr/bin"]
    job_python_first_on_path()
    assert os.environ["PATH"].split(os.pathsep)[0] == here
