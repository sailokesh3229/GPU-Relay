from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from gpurelay.accounts.sessions import SessionSigner
from gpurelay.accounts.store import UserStore
from gpurelay.accounts.vault import Vault, VaultError, new_key
from gpurelay.config import load_config, save_config
from gpurelay.errors import ConfigError
from gpurelay.gateway.app import GatewaySettings, create_gateway
from gpurelay.gateway.space import WORKER, UserSpaces, run_process, scrub
from gpurelay.gateway.userenv import kaggle_env, user_environment

PW = "correct horse battery"
HF_TOKEN = "hf_" + "a" * 30
KAGGLE_TOKEN = "KGAT_" + "b" * 30


def test_vault_seals_per_user_and_name() -> None:
    vault = Vault.from_env({"GPURELAY_VAULT_KEY": new_key()})
    sealed = vault.seal("u-1", "hf_token", HF_TOKEN)
    assert HF_TOKEN not in sealed and sealed.startswith("v1:")
    assert vault.open("u-1", "hf_token", sealed) == HF_TOKEN
    for user, name in (("u-2", "hf_token"), ("u-1", "kaggle_token")):  # moved rows fail
        with pytest.raises(VaultError):
            vault.open(user, name, sealed)
    with pytest.raises(VaultError):
        Vault.from_env({"GPURELAY_VAULT_KEY": new_key()}).open("u-1", "hf_token", sealed)
    for bad in ({}, {"GPURELAY_VAULT_KEY": "not base64!"}, {"GPURELAY_VAULT_KEY": "c2hvcnQ="}):
        with pytest.raises(ConfigError):
            Vault.from_env(bad)
    generated = {"GPURELAY_VAULT_KEY": "r4nd0m-value-the-host-generated-for-us-123"}
    sealed2 = Vault.from_env(generated).seal("u-1", "hf_token", HF_TOKEN)
    assert Vault.from_env(generated).open("u-1", "hf_token", sealed2) == HF_TOKEN  # stable


def test_user_environment_holds_only_that_users_tokens(tmp_path: Path) -> None:
    server = {"PATH": "/bin", "DATABASE_URL": "postgresql://x", "GPURELAY_VAULT_KEY": "k",
              "GPURELAY_SESSION_KEY": "s", "HF_TOKEN": "hf_server", "KAGGLE_API_TOKEN": "srv",
              "MODAL_TOKEN_ID": "srv", "GPURELAY_OWNER_PASSWORD": "p"}  # fmt: skip
    env = user_environment(tmp_path, {"hf_token": HF_TOKEN, "kaggle_token": KAGGLE_TOKEN}, server)
    assert env["PATH"] == "/bin" and env["HF_TOKEN"] == HF_TOKEN
    assert env["KAGGLE_API_TOKEN"] == KAGGLE_TOKEN and "MODAL_TOKEN_ID" not in env
    for name in ("DATABASE_URL", "GPURELAY_VAULT_KEY", "GPURELAY_SESSION_KEY",
                 "GPURELAY_OWNER_PASSWORD"):  # fmt: skip
        assert name not in env
    assert env["HOME"] == env["GPURELAY_HOME"] == str(tmp_path)
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert kaggle_env('{"username": "ada", "key": "abc"}') == {
        "KAGGLE_USERNAME": "ada", "KAGGLE_KEY": "abc"}  # fmt: skip


def test_errors_never_quote_a_token() -> None:
    assert scrub(f"401 for token {HF_TOKEN}", {"hf_token": HF_TOKEN}) == "401 for token ***"


def fake_worker(argv: list[str], env: dict[str, str], stdin: str, timeout: float) -> str:
    """Stands in for setup_worker: checks the tokens it was given, updates the config."""
    request = json.loads(stdin)
    home = Path(env["GPURELAY_HOME"])
    config = load_config(home / "config.toml")
    step = request["step"]
    fake_worker.calls.append((step, dict(env), request))  # type: ignore[attr-defined]
    if step == "hf":
        if env.get("HF_TOKEN") != HF_TOKEN:
            return json.dumps({"error": f"Hugging Face did not accept {env.get('HF_TOKEN')}"})
        config.storage.kind, config.storage.hf_repo_id = "hf_hub", "ada/gpurelay-store"
    elif step == "kaggle":
        if env.get("KAGGLE_API_TOKEN") != KAGGLE_TOKEN:
            return json.dumps({"error": "Kaggle did not accept this token"})
        config.backends.kaggle.enabled, config.backends.kaggle.username = True, "ada"
        if request.get("accelerator"):
            config.backends.kaggle.accelerator = request["accelerator"]
    elif step == "modal":
        config.backends.modal.terms_reviewed = True
        config.backends.modal.enabled = True
    save_config(config, home / "config.toml")
    return "noise\n" + json.dumps({"ok": True, "step": step})


fake_worker.calls = []  # type: ignore[attr-defined]


@pytest.fixture
def site(tmp_path: Path) -> tuple[TestClient, UserSpaces, list[str]]:
    store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
    vault = Vault.from_env({"GPURELAY_VAULT_KEY": new_key()})
    spaces = UserSpaces(store, vault, tmp_path / "users", runner=fake_worker)
    changed: list[str] = []
    settings = GatewaySettings(store=store, signer=SessionSigner(b"k" * 32), spaces=spaces,
                               secure_cookie=False, after_change=changed.append)  # fmt: skip
    client = TestClient(create_gateway(settings), follow_redirects=False)
    client.post(
        "/signup",
        data={"username": "ada", "email": "ada@example.com", "password": PW, "confirm": PW},
    )
    fake_worker.calls.clear()  # type: ignore[attr-defined]
    return client, spaces, changed


def test_setup_needs_a_signed_in_user(tmp_path: Path) -> None:
    store = UserStore(f"sqlite:///{tmp_path / 'u.db'}")
    spaces = UserSpaces(store, Vault.from_env({"GPURELAY_VAULT_KEY": new_key()}), tmp_path)
    client = TestClient(
        create_gateway(
            GatewaySettings(
                store=store, signer=SessionSigner(b"k" * 32), spaces=spaces, secure_cookie=False
            )
        )
    )
    assert client.get("/api/setup").status_code == 401
    assert client.post("/api/setup/hf", json={"token": HF_TOKEN}).status_code == 401


def test_full_setup_keeps_tokens_sealed(site: Any) -> None:
    client, spaces, changed = site
    assert client.get("/api/setup").json()["ready"] is False
    assert client.post("/api/setup/hf", json={"token": HF_TOKEN}).json()["ok"]
    response = client.post(
        "/api/setup/kaggle", json={"token": KAGGLE_TOKEN, "accelerator": "NvidiaTeslaP100"}
    )
    assert response.json()["ok"]
    kaggle_request = fake_worker.calls[-1][2]  # type: ignore[attr-defined]
    assert kaggle_request["rotate"] is True  # the secrets dataset gets this site's HF token
    status = client.get("/api/setup").json()
    assert status["ready"] and status["steps"]["hf"]["repo"] == "ada/gpurelay-store"
    assert status["steps"]["kaggle"] | {"saved_at": 0} == {
        "connected": True, "username": "ada", "accelerator": "NvidiaTeslaP100",
        "weekly_hours": 30.0, "saved_at": 0}  # fmt: skip
    assert HF_TOKEN not in json.dumps(status) and KAGGLE_TOKEN not in json.dumps(status)
    user = spaces.store.find("ada@example.com")
    raw = spaces.store.secret(user.id, "hf_token")
    assert raw and HF_TOKEN not in raw and spaces.secrets(user.id)["hf_token"] == HF_TOKEN
    assert HF_TOKEN not in (spaces.store.settings(user.id) or "")
    assert changed == [user.id, user.id]
    config = spaces.config(user.id)
    assert not config.backends.local.enabled  # never jobs on the server itself


def test_a_rejected_token_is_not_kept(site: Any) -> None:
    client, spaces, changed = site
    bad = "hf_" + "z" * 30
    response = client.post("/api/setup/hf", json={"token": bad})
    assert response.status_code == 400 and bad not in response.text  # scrubbed
    assert "***" in response.json()["detail"]
    user = spaces.store.find("ada@example.com")
    assert spaces.saved(user.id) == {} and changed == []
    assert client.post("/api/setup/hf", json={"token": "not-hf"}).status_code == 400
    assert client.post("/api/setup/kaggle", json={}).status_code == 400  # Kaggle before HF token


def test_a_new_hugging_face_token_refreshes_kaggle(site: Any) -> None:
    client, _, _ = site
    client.post("/api/setup/hf", json={"token": HF_TOKEN})
    client.post("/api/setup/kaggle", json={"token": KAGGLE_TOKEN})
    fake_worker.calls.clear()  # type: ignore[attr-defined]
    assert client.post("/api/setup/hf", json={"token": HF_TOKEN}).json()["ok"]
    steps = [(step, req.get("rotate")) for step, _, req in fake_worker.calls]  # type: ignore[attr-defined]
    assert steps == [("hf", None), ("kaggle", True)]
    kaggle_env_seen = fake_worker.calls[1][1]  # type: ignore[attr-defined]
    assert kaggle_env_seen["KAGGLE_API_TOKEN"] == KAGGLE_TOKEN  # the stored token is reused


def test_modal_needs_the_terms_and_colab_limits_remove(site: Any) -> None:
    client, spaces, _ = site
    client.post("/api/setup/hf", json={"token": HF_TOKEN})
    form = {"token_id": "ak-123456", "token_secret": "as-123456"}
    assert client.post("/api/setup/modal", json=form).status_code == 400
    assert client.post("/api/setup/modal", json={**form, "terms": True}).json()["ok"]
    assert client.post(
        "/api/setup/colab", json={"enabled": True, "first": True, "wait_min": 45}
    ).json()["ok"]
    assert client.post("/api/setup/limits", json={"modal_paid_cap_usd": 5}).json()["ok"]
    steps = client.get("/api/setup").json()["steps"]
    assert steps["modal"]["connected"] and steps["modal"]["paid_cap_usd"] == 5
    assert steps["colab"] == {"enabled": True, "first": True, "wait_min": 45}
    client.post("/api/setup/remove", json={"name": "modal"})
    assert not client.get("/api/setup").json()["steps"]["modal"]["connected"]
    user = spaces.store.find("ada@example.com")
    assert set(spaces.saved(user.id)) == {"hf_token"}
    client.post("/api/setup/remove", json={"name": "hf"})
    assert spaces.saved(user.id) == {}
    assert client.post("/api/setup/remove", json={"name": "db"}).status_code == 422


def test_users_never_see_each_others_setup(site: Any) -> None:
    client, _, _ = site
    client.post("/api/setup/hf", json={"token": HF_TOKEN})
    bob = TestClient(client.app, follow_redirects=False)
    bob.post(
        "/signup",
        data={"username": "bob", "email": "bob@example.com", "password": PW, "confirm": PW},
    )
    assert bob.get("/api/setup").json()["steps"]["hf"]["connected"] is False


def test_the_real_worker_process_answers_in_json(tmp_path: Path) -> None:
    env = user_environment(tmp_path, {})
    out = run_process(WORKER, env, json.dumps({"step": "hf"}), 120)
    assert json.loads(out.strip().splitlines()[-1]) == {
        "error": "a Hugging Face token starts with hf_"}  # fmt: skip
