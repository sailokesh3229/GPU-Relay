"""The website's setup worker (plan M2), which runs the real token checks in its own process:
each step with fake platforms, and errors that never quote a token."""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from gpurelay.config import Config, load_config, save_config
from gpurelay.gateway import setup_worker

HF_TOKEN = "hf_" + "w" * 30


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GPURELAY_HOME", str(tmp_path))
    save_config(Config(), tmp_path / "config.toml")
    return tmp_path


def _run(monkeypatch: pytest.MonkeyPatch, request: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert setup_worker.main() == 0
    return dict(json.loads(out.getvalue().strip().splitlines()[-1]))


class FakeHfApi:
    accept = True

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def whoami(self) -> dict[str, str]:
        if not self.accept:
            raise RuntimeError("401 Unauthorized")
        return {"name": "ada"}


def test_unknown_steps_and_broken_requests_answer_with_an_error(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(monkeypatch, {"step": "nope"}) == {"error": "unknown setup step"}
    monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    setup_worker.main()
    assert json.loads(out.getvalue().strip()) == {"error": "setup failed (JSONDecodeError)"}


def test_hugging_face_step_checks_the_token_then_sets_up_the_repo(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    import gpurelay.controller.setup as setup_mod

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setenv("HF_TOKEN", "not-a-token")
    assert "starts with hf_" in _run(monkeypatch, {"step": "hf"})["error"]

    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)
    FakeHfApi.accept = False
    assert "did not accept" in _run(monkeypatch, {"step": "hf"})["error"]
    FakeHfApi.accept = True

    def setup_storage(config: Config, api: Any, repo: str | None = None) -> Any:
        config.storage.kind, config.storage.hf_repo_id = "hf_hub", repo or "ada/gpurelay-store"
        return types.SimpleNamespace(hf_user="ada", repo_id=config.storage.hf_repo_id,
                                     created=True)  # fmt: skip

    monkeypatch.setattr(setup_mod, "setup_storage", setup_storage)
    answer = _run(monkeypatch, {"step": "hf", "repo": " ada/my-store "})
    assert answer == {"ok": True, "user": "ada", "repo": "ada/my-store", "created": True}
    assert load_config(home / "config.toml").storage.hf_repo_id == "ada/my-store"

    def no_write(config: Config, api: Any, repo: str | None = None) -> Any:
        raise PermissionError("403 read-only token")

    monkeypatch.setattr(setup_mod, "setup_storage", no_write)
    assert "write access" in _run(monkeypatch, {"step": "hf"})["error"]


def test_kaggle_and_modal_need_storage_first(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, {"step": "kaggle"})["error"] == "connect Hugging Face first"
    assert _run(monkeypatch, {"step": "modal", "terms": True})["error"] == (
        "connect Hugging Face first")  # fmt: skip


def _with_storage(home: Path) -> None:
    config = load_config(home / "config.toml")
    config.storage.kind, config.storage.hf_repo_id = "hf_hub", "ada/gpurelay-store"
    save_config(config, home / "config.toml")


def test_modal_step_checks_terms_and_the_token_then_turns_modal_on(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_storage(home)
    assert "terms" in _run(monkeypatch, {"step": "modal"})["error"]
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-1")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-2")
    calls: list[tuple[str, str]] = []

    class Client:
        refuse = False

        @classmethod
        def from_credentials(cls, token_id: str, token_secret: str) -> Client:
            calls.append((token_id, token_secret))
            return cls()

        def hello(self) -> None:
            if self.refuse:
                raise RuntimeError("bad token")

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Client=Client))
    answer = _run(monkeypatch, {"step": "modal", "terms": True, "gpu": "L4"})
    assert answer == {"ok": True, "gpu": "L4"} and calls == [("ak-1", "as-2")]
    modal = load_config(home / "config.toml").backends.modal
    assert modal.enabled and modal.terms_reviewed and modal.gpu == "L4"
    Client.refuse = True
    assert "did not accept" in _run(monkeypatch, {"step": "modal", "terms": True})["error"]


def test_kaggle_step_sets_up_the_secrets_dataset(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    import gpurelay.backends.kaggle as kaggle_mod
    import gpurelay.controller.setup as setup_mod

    _with_storage(home)
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(setup_mod, "validate_session_token", lambda token, repo, api: None)
    monkeypatch.setattr(kaggle_mod, "OfficialKaggleClient", lambda: object())
    seen: dict[str, Any] = {}

    def setup_kaggle(config: Config, client: Any, *, session_token: str, repo_id: str,
                     rotate: bool) -> Any:  # fmt: skip
        seen.update(token=session_token, repo=repo_id, rotate=rotate)
        config.backends.kaggle.enabled, config.backends.kaggle.username = True, "ada"
        return types.SimpleNamespace(username="ada", created=False)

    monkeypatch.setattr(setup_mod, "setup_kaggle", setup_kaggle)
    answer = _run(monkeypatch, {"step": "kaggle", "rotate": True, "accelerator": "NvidiaTeslaP100"})
    assert answer == {"ok": True, "username": "ada", "created": False}
    assert seen == {"token": HF_TOKEN, "repo": "ada/gpurelay-store", "rotate": True}
    kaggle = load_config(home / "config.toml").backends.kaggle
    assert kaggle.enabled and kaggle.accelerator == "NvidiaTeslaP100"


def test_an_unexpected_error_never_reaches_the_page_with_its_text(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"connection reset while sending {HF_TOKEN}")

    monkeypatch.setitem(setup_worker.STEPS, "hf", boom)
    answer = _run(monkeypatch, {"step": "hf"})
    assert answer == {"error": "setup failed (RuntimeError)"} and HF_TOKEN not in json.dumps(answer)
