"""Your GPU platforms (spec §21.1, plan G1): asked once at first sign-in, changed any time
from Settings, and nothing saved before is ever lost. Website (vault) and desktop (keyring)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography")
from fastapi.testclient import TestClient

from gpurelay.accounts.sessions import SessionSigner
from gpurelay.accounts.store import UserStore
from gpurelay.accounts.vault import Vault, new_key
from gpurelay.dashboard.desktop_api import SetupDeps, create_setup_app
from gpurelay.gateway.app import GatewaySettings, create_gateway
from gpurelay.gateway.space import UserSpaces
from gpurelay.gateway.userenv import user_environment
from gpurelay.platforms.catalog import BY_ID, PLATFORMS, SECRET_PREFIX, SECTIONS, secret_name

PW = "correct horse battery"
HF_TOKEN = "hf_" + "a" * 30
RUNPOD_KEY = "rpa_" + "r" * 40
SA_KEY = json.dumps({"type": "service_account", "project_id": "p", "private_key": "-----BEGIN-----",
                     "client_email": "relay@p.iam.gserviceaccount.com"})  # fmt: skip


def test_the_catalog_has_three_sections_and_distinct_platforms() -> None:
    sections = {s for s, _ in SECTIONS}
    assert sections == {p.section for p in PLATFORMS}
    assert len(BY_ID) == len(PLATFORMS)
    assert {p.id for p in PLATFORMS if p.section == "hyperscaler"} == {"gcp", "aws", "azure"}
    assert {p.id for p in PLATFORMS if p.section == "free"} == {"colab", "kaggle", "modal"}
    assert {"runpod", "vastai", "lambda", "coreweave", "digitalocean"} <= {
        p.id for p in PLATFORMS if p.section == "gpu_cloud"}  # fmt: skip
    for p in PLATFORMS:  # free platforms connect on the setup page, never here
        assert (p.support == "setup") == (p.section == "free") and (not p.fields or p.how)


# ---------------------------------------------------------------- website


def _site(tmp_path: Path) -> tuple[TestClient, UserSpaces]:
    store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
    spaces = UserSpaces(store, Vault.from_env({"GPURELAY_VAULT_KEY": new_key()}), tmp_path / "u")
    settings = GatewaySettings(store=store, signer=SessionSigner(b"k" * 32), spaces=spaces,
                               secure_cookie=False)  # fmt: skip
    client = TestClient(create_gateway(settings), follow_redirects=False)
    return client, spaces


@pytest.fixture
def site(tmp_path: Path) -> tuple[TestClient, UserSpaces, str]:
    client, spaces = _site(tmp_path)
    response = client.post("/signup", data={"username": "ada", "email": "ada@example.com",
                                            "password": PW, "confirm": PW})  # fmt: skip
    assert response.headers["location"] == "/platforms"  # the first thing a new user sees
    return client, spaces, spaces.store.find("ada@example.com").id


def _platform(view: dict[str, Any], platform_id: str) -> dict[str, Any]:
    return next(p for p in view["platforms"] if p["id"] == platform_id)


def test_the_page_and_its_api_need_a_signed_in_user(tmp_path: Path) -> None:
    client, _ = _site(tmp_path)
    assert client.get("/platforms").headers["location"] == "/signin"
    assert client.get("/api/platforms").status_code == 401
    assert client.post("/api/platforms", json={"done": True}).status_code == 401


def test_asked_once_then_setup_and_skipping_is_allowed(site: Any) -> None:
    client, _, _ = site
    assert "Your GPU platforms" in client.get("/platforms").text
    view = client.get("/api/platforms").json()
    assert view["asked"] is False and [s["id"] for s in view["sections"]] == [
        "hyperscaler", "gpu_cloud", "free"]  # fmt: skip
    skipped = client.post("/api/platforms", json={"done": True}).json()
    assert skipped["asked"] is True and skipped["next"] == "/setup"
    assert client.get("/").headers["location"] == "/setup"
    again = TestClient(client.app, follow_redirects=False)  # signing in later: not asked again
    assert again.post("/signin", data={"login": "ada", "password": PW}).headers[
        "location"] == "/setup"  # fmt: skip


def test_answers_and_sealed_credentials_never_lost_and_never_given_to_jobs(site: Any) -> None:
    client, spaces, uid = site
    spaces.put_secrets(uid, {"hf_token": HF_TOKEN})  # a job token from setup
    first = client.post("/api/platforms", json={
        "answers": {"aws": "no", "gcp": "yes", "kaggle": "yes", "azure": "later"},
        "credentials": {"runpod": {"api_key": RUNPOD_KEY},
                        "gcp": {"project_id": "my-proj", "service_account_key": SA_KEY}},
        "others": ["TensorDock", " tensordock ", "Jarvis  Labs"], "done": True,
    })  # fmt: skip
    assert first.status_code == 200, first.text
    view = first.json()
    assert RUNPOD_KEY not in first.text and "private_key" not in first.text  # never returned
    assert _platform(view, "runpod")["has"] == "yes"  # a key means an account
    assert all(f["saved_at"] for f in _platform(view, "gcp")["fields"])
    assert view["others"] == ["TensorDock", "Jarvis Labs"]
    raw = spaces.store.secret(uid, secret_name("runpod", "api_key"))
    assert raw and RUNPOD_KEY not in raw  # sealed in the database
    # no process acting for the user gets them: job tokens only
    env = user_environment(Path("h"), spaces.secrets(uid), base={})
    assert RUNPOD_KEY not in json.dumps(env) and env["HF_TOKEN"] == HF_TOKEN

    # Settings, later: bought Vast.ai credits. Only what is named changes.
    later = client.post("/api/platforms", json={
        "answers": {"aws": "yes"}, "credentials": {"vastai": {"api_key": "v" * 64},
                                                   "runpod": {"api_key": "  "}}})  # fmt: skip
    view = later.json()
    assert _platform(view, "aws")["has"] == "yes" and _platform(view, "gcp")["has"] == "yes"
    assert _platform(view, "azure")["has"] == "later" and _platform(view, "vastai")["has"] == "yes"
    assert view["others"] == ["TensorDock", "Jarvis Labs"]  # not sent: kept
    saved = spaces.saved(uid)
    assert {secret_name("runpod", "api_key"), secret_name("vastai", "api_key"),
            secret_name("gcp", "service_account_key"), "hf_token"} <= set(saved)  # fmt: skip

    # Removing RunPod's key removes only that; answers and every other credential stay.
    removed = client.post("/api/platforms/remove", json={"platform": "runpod"}).json()
    assert _platform(removed, "runpod")["has"] == "yes"
    assert not _platform(removed, "runpod")["fields"][0]["saved_at"]
    left = set(spaces.saved(uid))
    assert secret_name("runpod", "api_key") not in left and "hf_token" in left
    assert secret_name("vastai", "api_key") in left


def test_other_setup_steps_keep_the_answers(site: Any) -> None:
    client, spaces, uid = site
    client.post("/api/platforms", json={"answers": {"aws": "yes"}, "done": True})
    config = spaces.config(uid)  # what a setup step does: read, change, save
    config.backends.colab_manual.enabled = True
    spaces.save_config(uid, config)
    assert client.post("/api/setup/colab", json={"enabled": False}).status_code in (200, 400)
    assert spaces.config(uid).platforms.answers["aws"].has == "yes"
    assert _platform(client.get("/api/platforms").json(), "aws")["has"] == "yes"


def test_bad_input_saves_nothing(site: Any) -> None:
    client, spaces, uid = site
    for body, error in (
        ({"answers": {"nimbus": "yes"}}, "unknown platform"),
        ({"credentials": {"runpod": {"password": "x"}}}, "no field"),
        ({"credentials": {"runpod": {"api_key": "a\nb"}}}, "one line"),
        ({"credentials": {"runpod": {"api_key": RUNPOD_KEY},
                          "gcp": {"service_account_key": '{"type": "user"}'}}},
         "not a service account key"),
        ({"credentials": {"gcp": {"service_account_key": "not json"}}}, "JSON"),
        ({"others": ["x" * 61]}, "at most"),
    ):  # fmt: skip
        response = client.post("/api/platforms", json=body)
        assert response.status_code == 400 and error in response.json()["detail"], body
    assert not [n for n in spaces.saved(uid) if n.startswith(SECRET_PREFIX)]
    assert spaces.config(uid).platforms.answers == {}
    assert client.post("/api/platforms", json={"answers": {"aws": "maybe"}}).status_code == 422


def test_free_platforms_link_to_setup_and_show_when_connected(site: Any) -> None:
    client, spaces, uid = site
    view = client.get("/api/platforms").json()
    assert _platform(view, "kaggle")["setup"] == "/setup#kaggle"
    assert not _platform(view, "colab")["connected"]
    config = spaces.config(uid)
    config.backends.colab_manual.enabled = True
    spaces.save_config(uid, config)
    assert _platform(client.get("/api/platforms").json(), "colab")["connected"]


def test_too_many_changes_are_refused(site: Any) -> None:
    client, _, _ = site
    codes = [client.post("/api/platforms", json={}).status_code for _ in range(61)]
    assert codes[:60] == [200] * 60 and codes[60] == 429


# ---------------------------------------------------------------- desktop

KEY = "launch-key-123"


class Hooks:
    def __init__(self, home: Path) -> None:
        self.home = home

    def controller_state(self) -> dict[str, Any]:
        return {"mode": "local", "running": False}

    def autostart(self) -> bool:
        return False

    def packaged(self) -> bool:
        return False

    def start_menu(self) -> bool:
        return False

    def update_info(self) -> dict[str, Any]:
        return {}

    def reconfigured(self) -> None:
        self.restarts = getattr(self, "restarts", 0) + 1


@pytest.fixture
def desktop(tmp_path: Path) -> tuple[TestClient, dict[str, str], Path]:
    keyring: dict[str, str] = {}
    deps = SetupDeps(
        get_secret=lambda name, env: keyring.get(name),
        set_secret=lambda name, value: keyring.__setitem__(name, value) is None,
        delete_secret=lambda name: keyring.pop(name, None) and None,
        kaggle_present=lambda: False,
        modal_present=lambda: False,
    )  # fmt: skip
    home = tmp_path / "home"
    client = TestClient(create_setup_app(Hooks(home), launch_key=KEY, deps=deps),  # type: ignore[arg-type]
                        follow_redirects=False)  # fmt: skip
    client.get(f"/?k={KEY}")
    return client, keyring, home


ACT = {"X-GPURelay-Action": "1"}


def test_desktop_asks_first_and_keeps_credentials_in_the_keyring(desktop: Any) -> None:
    client, keyring, home = desktop
    assert client.get("/api/desktop/status").json()["platforms_asked"] is False
    assert "Your GPU platforms" in client.get("/platforms").text
    body = {"answers": {"aws": "yes"}, "credentials": {"lambda": {"api_key": "L" * 40}},
            "done": True}  # fmt: skip
    assert client.post("/api/platforms", json=body).status_code == 403  # needs the header
    view = client.post("/api/platforms", json=body, headers=ACT).json()
    assert view["next"] == "./#/setup" and _platform(view, "lambda")["fields"][0]["saved_at"]
    assert keyring == {secret_name("lambda", "api_key"): "L" * 40}
    assert "L" * 40 not in (home / "platforms.json").read_text()
    assert not (home / "config.toml").exists()  # setup is not marked done by answering
    status = client.get("/api/desktop/status").json()
    assert status["platforms_asked"] is True and status["configured"] is False
    client.post("/api/platforms", json={"answers": {"gcp": "no"}}, headers=ACT)
    view = client.get("/api/platforms").json()
    assert _platform(view, "aws")["has"] == "yes" and _platform(view, "gcp")["has"] == "no"
    removed = client.post("/api/platforms/remove", json={"platform": "lambda"}, headers=ACT)
    assert removed.status_code == 200 and keyring == {}
    assert _platform(removed.json(), "lambda")["has"] == "yes"


def test_desktop_never_overwrites_answers_it_cannot_read(desktop: Any) -> None:
    client, _, home = desktop
    home.mkdir(parents=True)
    (home / "platforms.json").write_text("{broken", encoding="utf-8")
    response = client.post("/api/platforms", json={"answers": {"aws": "yes"}}, headers=ACT)
    assert response.status_code == 400 and "damaged" in response.json()["detail"]
    assert (home / "platforms.json").read_text(encoding="utf-8") == "{broken"
    assert client.get("/api/desktop/status").json()["platforms_asked"] is True  # not re-asked


def test_desktop_without_a_keyring_saves_no_credentials(tmp_path: Path) -> None:
    deps = SetupDeps(get_secret=lambda n, e: None, set_secret=lambda n, v: False,
                     delete_secret=lambda n: None)  # fmt: skip
    client = TestClient(create_setup_app(Hooks(tmp_path), launch_key=KEY, deps=deps),  # type: ignore[arg-type]
                        follow_redirects=False)  # fmt: skip
    client.get(f"/?k={KEY}")
    response = client.post("/api/platforms", headers=ACT,
                           json={"credentials": {"runpod": {"api_key": RUNPOD_KEY}}})  # fmt: skip
    assert response.status_code == 400 and "keyring" in response.json()["detail"]
    assert not _platform(client.get("/api/platforms").json(), "runpod")["has"]


# ---------------------------------------------------------------- turning a paid platform on


def test_turning_a_paid_platform_on_needs_its_terms_and_key_then_restarts(tmp_path: Path) -> None:
    store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
    spaces = UserSpaces(store, Vault.from_env({"GPURELAY_VAULT_KEY": new_key()}), tmp_path / "u")
    changed: list[str] = []
    settings = GatewaySettings(store=store, signer=SessionSigner(b"k" * 32), spaces=spaces,
                               secure_cookie=False, after_change=changed.append)  # fmt: skip
    client = TestClient(create_gateway(settings), follow_redirects=False)
    client.post("/signup", data={"username": "ada", "email": "ada@example.com",
                                 "password": PW, "confirm": PW})  # fmt: skip
    uid = store.find("ada@example.com").id
    on = {"use": {"runpod": {"enabled": True}}}
    response = client.post("/api/platforms", json=on)
    assert response.status_code == 400 and "terms" in response.json()["detail"]
    response = client.post(
        "/api/platforms", json={"use": {"runpod": {"enabled": True, "terms": True}}}
    )
    assert response.status_code == 400 and "API key" in response.json()["detail"]
    assert not spaces.config(uid).backends.runpod.enabled and changed == []
    assert spaces.platform_env(uid) == {}  # a saved key alone gives nothing to any process
    client.post("/api/platforms", json={"credentials": {"runpod": {"api_key": RUNPOD_KEY}}})
    assert spaces.platform_env(uid) == {}  # still off
    response = client.post("/api/platforms", json={"use": {"runpod": {
        "enabled": True, "terms": True, "credit_usd": 25.5, "card_limit_usd": 0,
        "machine": "NVIDIA RTX A5000"}}})  # fmt: skip
    assert response.status_code == 200, response.text
    runpod = spaces.config(uid).backends.runpod
    assert runpod.enabled and runpod.credit_usd == 25.5 and runpod.credit_set_at
    assert runpod.gpu == "NVIDIA RTX A5000" and changed == [uid]  # their GPU Relay restarts
    assert spaces.platform_env(uid) == {"RUNPOD_API_KEY": RUNPOD_KEY}
    use = _platform(response.json(), "runpod")["use"]
    assert use["enabled"] and use["credit_usd"] == 25.5 and RUNPOD_KEY not in response.text
    bad = client.post("/api/platforms", json={"use": {"runpod": {"machine": "rm -rf /;"}}})
    assert bad.status_code == 400
    assert client.post("/api/platforms", json={"use": {"coreweave": {"enabled": True}}}
                       ).status_code == 400  # fmt: skip
    # Removing the key turns the platform off, so nothing tries to use it.
    client.post("/api/platforms/remove", json={"platform": "runpod"})
    assert not spaces.config(uid).backends.runpod.enabled and spaces.platform_env(uid) == {}


def test_desktop_turns_platforms_on_only_after_setup(desktop: Any) -> None:
    client, keyring, home = desktop
    assert client.get("/api/platforms").json()["can_use"] is False
    body = {"credentials": {"vastai": {"api_key": "v" * 40}},
            "use": {"vastai": {"enabled": True, "terms": True}}}  # fmt: skip
    response = client.post("/api/platforms", json=body, headers=ACT)
    assert response.status_code == 400 and "finish setting up" in response.json()["detail"]
    assert keyring == {}  # nothing kept from a refused save
    from gpurelay.config import Config, load_config, save_config

    save_config(Config(), home / "config.toml")
    response = client.post("/api/platforms", json=body, headers=ACT)
    assert response.status_code == 200, response.text
    assert load_config(home / "config.toml").backends.vastai.enabled
    assert keyring == {secret_name("vastai", "api_key"): "v" * 40}
