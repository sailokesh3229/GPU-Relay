"""Notifications (spec §13.6): desktop and the ntfy.sh topic; a failure never stops the
controller."""

from __future__ import annotations

import json
import sys
import types
import urllib.request
from typing import Any

import pytest

from gpurelay.config import NotifySettings
from gpurelay.controller.notify import NTFY_URL, Notifier


def test_ntfy_carries_any_text_in_a_json_body() -> None:
    posts: list[tuple[str, bytes, dict[str, str]]] = []

    def post(url: str, body: bytes, headers: dict[str, str]) -> None:
        posts.append((url, body, headers))

    notifier = Notifier(NotifySettings(desktop=False, ntfy_topic="ada-jobs"), _post=post)
    notifier.notify("completed", "job శిక్షణ 🚀 completed", "1 segment(s)")
    [(url, body, headers)] = posts
    assert url == NTFY_URL and headers == {"Content-Type": "application/json"}
    assert json.loads(body) == {"topic": "ada-jobs", "title": "GPU Relay: job శిక్షణ 🚀 completed",
                                "message": "1 segment(s)", "tags": ["completed"]}  # fmt: skip


def test_a_real_http_post_of_a_non_latin_title_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    class Answer:
        def __enter__(self) -> Answer:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    def urlopen(request: Any, timeout: float) -> Answer:
        for _name, value in request.header_items():  # what http.client would refuse
            value.encode("latin-1")
        sent.append(request)
        return Answer()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    Notifier(NotifySettings(desktop=False, ntfy_topic="t")).notify("failed", "जॉब failed", "x")
    assert len(sent) == 1


def test_delivery_failures_are_logged_never_raised(caplog: pytest.LogCaptureFixture) -> None:
    def down(url: str, body: bytes, headers: dict[str, str]) -> None:
        raise OSError("network unreachable")

    notifier = Notifier(NotifySettings(desktop=False, ntfy_topic="t"), _post=down)
    with caplog.at_level("WARNING"):
        notifier.notify("budget_low", "kaggle budget low", "1.2 GPU-h left")
    assert "ntfy notification failed" in caplog.text and len(notifier.sent) == 1


def test_desktop_notifications_go_through_plyer(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[dict[str, str]] = []
    plyer = types.ModuleType("plyer")
    plyer.notification = types.SimpleNamespace(notify=lambda **kw: shown.append(kw))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "plyer", plyer)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    Notifier(NotifySettings(desktop=True)).notify("colab_action", "job vae needs you", "m" * 400)
    [note] = shown
    assert note["title"] == "GPU Relay: job vae needs you" and len(note["message"]) == 250
