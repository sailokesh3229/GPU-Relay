"""Web app write screens (plan W2): analyze, submit, output files, budget correction, settings."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from test_controller import Harness, _harness

from gpurelay.controller.db import LedgerEntry
from gpurelay.controller.service import Context
from gpurelay.dashboard.app import ACTION_HEADER, LOCAL_HOSTS, create_app
from gpurelay.dashboard.forms import MAX_BODY_BYTES, daily_usage
from gpurelay.models import JobState
from gpurelay.storage.layout import bundle_prefix, final_prefix

ACT = {ACTION_HEADER: "1"}
SCRIPT = (
    b"import time\nSTEPS = 10\nfor step in range(STEPS):\n    time.sleep(0.01)\nprint('done')\n"
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _setup(tmp_path: Path) -> tuple[TestClient, Harness]:
    h = _harness(tmp_path, {"kaggle": (30.0, True, 1, "T4")})
    ctx = Context(h.controller.config, tmp_path / "home", h.db, h.storage, h.controller)
    app = create_app(ctx, clock=h.clock, allowed_hosts=LOCAL_HOSTS | {"testserver"})
    return TestClient(app), h


def test_analyze_an_upload(tmp_path: Path) -> None:
    client, _ = _setup(tmp_path)
    r = client.post("/api/analyze", json={"filename": "train.py", "content": _b64(SCRIPT)},
                    headers=ACT)  # fmt: skip
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["lines"] == 5 and data["analysis"]["overall"] == "FULL"
    assert data["analysis"]["blocks"][0]["loops"][0]["header"] == "for step in range(STEPS)"


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"filename": "../evil.py", "content": _b64(SCRIPT)}, "plain name"),
        ({"filename": "data.csv", "content": _b64(SCRIPT)}, ".py"),
        ({"filename": "t.py", "content": "not base64!"}, "base64"),
        ({"filename": "t.py", "content": ""}, "empty"),
        ({"filename": "t.py", "content": _b64(b"\xff\xfe")}, "UTF-8"),
        ({"filename": "t.py", "content": _b64(b"def broken(:\n")}, "syntax error"),
        ({"filename": "t.py", "content": _b64(SCRIPT), "extra": 1}, "extra"),
    ],
)
def test_bad_uploads_get_a_clear_400(tmp_path: Path, body: dict[str, object], detail: str) -> None:
    client, _ = _setup(tmp_path)
    r = client.post("/api/analyze", json=body, headers=ACT)
    assert r.status_code == 400 and detail.lower() in r.json()["detail"].lower()


def test_uploads_need_the_action_header_and_have_a_size_limit(tmp_path: Path) -> None:
    client, _ = _setup(tmp_path)
    body = {"filename": "t.py", "content": _b64(SCRIPT)}
    assert client.post("/api/analyze", json=body).status_code == 403
    assert client.post("/api/submit", json=body).status_code == 403
    huge = client.post("/api/analyze", content=b"x" * (MAX_BODY_BYTES + 1), headers=ACT)
    assert huge.status_code == 400 and "too large" in huge.json()["detail"]


def test_submit_queues_a_job_with_its_options(tmp_path: Path) -> None:
    client, h = _setup(tmp_path)
    r = client.post("/api/submit", headers=ACT, json={
        "filename": "train.py", "content": _b64(SCRIPT), "name": " my run ", "priority": 3,
        "estimate": "2h", "prefer": "kaggle", "data": ["hf:me/data", " "],
        "requirements": "numpy\n",
    })  # fmt: skip
    assert r.status_code == 200, r.text
    job = h.job(r.json()["job_id"])
    assert (job.name, job.priority, job.prefer_backend, job.estimate_h) == (
        "my run",
        3,
        "kaggle",
        2.0,
    )
    assert job.state is JobState.QUEUED and job.entry == "train.py"
    assert [d.ref for d in job.data] == ["me/data"]
    assert h.storage.list(bundle_prefix(job.id))  # the bundle was uploaded
    detail = client.get(f"/api/jobs/{job.id}").json()
    assert detail["job"]["entry"] == "train.py" and detail["checkpoints"] == 0
    assert detail["last_checkpoint"] is None


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        ({"data": ["local:C:/data"]}, "local:"),
        ({"prefer": "colab"}, "unknown backend"),
        ({"estimate": "soon"}, ""),
        ({"priority": 5000}, "priority"),
    ],
)
def test_submit_rejects_bad_options(tmp_path: Path, extra: dict[str, object], detail: str) -> None:
    client, h = _setup(tmp_path)
    body = {"filename": "train.py", "content": _b64(SCRIPT), **extra}
    r = client.post("/api/submit", json=body, headers=ACT)
    assert r.status_code == 400 and detail in r.json()["detail"]
    assert h.db.list_jobs() == []


def test_restart_only_jobs_need_explicit_consent(tmp_path: Path) -> None:
    client, h = _setup(tmp_path)
    main_pattern = b"def main():\n    for i in range(10):\n        pass\n\nmain()\n"
    body = {"filename": "wrap.py", "content": _b64(main_pattern)}
    analysis = client.post("/api/analyze", json=body, headers=ACT).json()["analysis"]
    assert analysis["overall"] == "RESTART_ONLY"
    r = client.post("/api/submit", json=body, headers=ACT)
    assert r.status_code == 400 and "no safe points" in r.json()["detail"]
    ok = client.post("/api/submit", json={**body, "accept_restart_only": True}, headers=ACT)
    assert ok.status_code == 200 and h.job(ok.json()["job_id"]).accept_restart_only


def test_output_files_list_and_download(tmp_path: Path) -> None:
    client, h = _setup(tmp_path)
    job = h.submit()
    assert client.get(f"/api/jobs/{job.id}/files").json() == {"completed": False, "files": []}
    h.storage.put_bytes(b"metric,1\n", f"{final_prefix(job.id)}/outputs/metrics.csv")
    h.storage.put_bytes(b"log\n", f"{final_prefix(job.id)}/log.txt")
    h.db.put_job(h.job(job.id).model_copy(update={"state": JobState.COMPLETED}))
    files = client.get(f"/api/jobs/{job.id}/files").json()
    assert files == {"completed": True, "files": ["log.txt", "outputs/metrics.csv"]}
    r = client.get(f"/api/jobs/{job.id}/files/outputs/metrics.csv")
    assert r.status_code == 200 and r.content == b"metric,1\n"
    assert "metrics.csv" in r.headers["content-disposition"]
    for bad in ("outputs%2F..%2Flog.txt", "..%2Fbundle%2Ftrain.py", "nope.txt"):
        assert client.get(f"/api/jobs/{job.id}/files/{bad}").status_code == 404


def test_budget_correction_and_daily_usage(tmp_path: Path) -> None:
    client, h = _setup(tmp_path)
    assert client.post("/api/quota/kaggle", json={"used": "12h"}).status_code == 403
    r = client.post("/api/quota/kaggle", json={"used": "12h"}, headers=ACT)
    assert r.status_code == 200
    [entry] = [e for e in h.db.ledger("kaggle") if e.kind == "correction"]
    assert entry.used_h == 12.0
    for body in ({"used": "1h", "remaining": "2h"}, {}, {"used": "lots"}):
        assert client.post("/api/quota/kaggle", json=body, headers=ACT).status_code == 400
    assert client.post("/api/quota/nope", json={"used": "1h"}, headers=ACT).status_code == 400
    [kaggle] = client.get("/api/overview").json()["backends"]
    assert len(kaggle["daily_h"]) == 7


def _usage(start: float, end: float, kind: str = "usage") -> LedgerEntry:
    return LedgerEntry(id=0, backend="k", kind=kind, start_ts=start, end_ts=end, hours=None,
                       remaining_h=None, used_h=None, segment_id=None, note=None,
                       created_ts=end)  # type: ignore[call-arg]  # fmt: skip


def test_daily_usage_splits_sessions_across_days() -> None:
    day = 86400.0
    now = 10 * day + 6 * 3600  # 06:00 on day 10
    entries = [
        _usage(9 * day - 3600, 9 * day + 3600),  # 23:00 day 8 -> 01:00 day 9
        _usage(10 * day, now + 3600),  # still running: counted up to now
        _usage(2 * day, 2 * day + 3600),  # older than the window
        _usage(9 * day, 9 * day + 7200, kind="correction"),
    ]
    out = daily_usage(entries, now)
    assert out == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 6.0]


def test_settings_report_status_not_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_example_value_for_test")
    client, _ = _setup(tmp_path)
    r = client.get("/api/settings")
    data = r.json()
    assert data["server"] is False and data["storage"]["token"] is True
    assert {b["name"] for b in data["backends"]} == {"kaggle", "colab", "lightning", "local"}
    assert "hf_example_value_for_test" not in r.text


def test_page_has_every_screen(tmp_path: Path) -> None:
    client, _ = _setup(tmp_path)
    page = client.get("/").text
    for marker in ("#/submit", "#/budget", "#/settings", "Resumability check", "Relay timeline"):
        assert marker in page
