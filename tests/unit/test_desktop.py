"""Desktop app (plan D2-D3): settings, single instance, notifications, autostart, icon, the
per-launch key, the setup wizard API and the app's reconfiguration. No window is opened."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from gpurelay.backends.kaggle import SECRETS_FILE
from gpurelay.config import Config, StorageSettings, load_config, save_config
from gpurelay.constants import CONFIG_FILENAME
from gpurelay.dashboard.desktop_api import (
    ACTION_HEADER,
    LAUNCH_COOKIE,
    SetupDeps,
    create_setup_app,
)
from gpurelay.desktop import autostart
from gpurelay.desktop.settings import (
    LOCAL,
    SERVER,
    DesktopSettings,
    InstanceLock,
    normalize_server_url,
)
from gpurelay.desktop.watcher import JobWatcher
from gpurelay.errors import ConfigError, GpuRelayError

ACT = {ACTION_HEADER: "1"}
KEY = "launch-key-123"
HF = "hf_" + "a" * 30  # fake tokens: never real values
KGAT = "KGAT_" + "b" * 30


# ---------------------------------------------------------------- settings and lock


def test_settings_round_trip_and_bad_files(tmp_path: Path) -> None:
    assert DesktopSettings.load(tmp_path) == DesktopSettings()
    DesktopSettings(mode=SERVER, server_url="https://x.onrender.com").save(tmp_path)
    assert DesktopSettings.load(tmp_path) == DesktopSettings(SERVER, "https://x.onrender.com")
    (tmp_path / "desktop.json").write_text('{"mode": "evil", "server_url": 5}')
    assert DesktopSettings.load(tmp_path) == DesktopSettings()
    (tmp_path / "desktop.json").write_text("not json")
    assert DesktopSettings.load(tmp_path) == DesktopSettings()


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("gpu-relay-9uuh.onrender.com", "https://gpu-relay-9uuh.onrender.com"),
        ("https://Gpu-Relay.onrender.com/", "https://gpu-relay.onrender.com"),
        ("http://localhost:7860", "http://localhost:7860"),
        ("https://example.com:8443", "https://example.com:8443"),
    ],
)
def test_server_urls_are_normalized(given: str, expected: str) -> None:
    assert normalize_server_url(given) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "ftp://x.com",
        "http://example.com",
        "https://u:p@x.com",
        "https://x.com/path",
        "https://x.com/?q=1",
    ],
)
def test_bad_server_urls_are_refused(bad: str) -> None:
    with pytest.raises(ConfigError):
        normalize_server_url(bad)


def test_only_one_desktop_app_runs_and_a_second_launch_asks_it_to_show(tmp_path: Path) -> None:
    first, second = InstanceLock(tmp_path), InstanceLock(tmp_path)
    assert first.acquire()
    assert not second.acquire()
    second.ask_running_app_to_show()
    assert first.show_requested() and not first.show_requested()
    first.release()
    assert second.acquire()
    second.release()


# ---------------------------------------------------------------- notifications


def _ov(*jobs: tuple[str, str]) -> dict[str, Any]:
    terminal = {"completed", "failed_user", "failed_infra", "cancelled"}
    return {"jobs": [{"id": i, "name": f"n-{i}", "state": s, "terminal": s in terminal,
                      "used_h": 0.5, "message": "boom"} for i, s in jobs]}  # fmt: skip


def test_watcher_primes_silently_then_reports_changes() -> None:
    w = JobWatcher()
    assert w.update(_ov(("a", "completed"), ("b", "running"))) == []  # no replay at startup
    assert w.summary() == "GPU Relay: 1 job active"
    notes = w.update(_ov(("a", "completed"), ("b", "awaiting_user"), ("c", "queued")))
    assert [(n.kind, n.job_id) for n in notes] == [("colab_action", "b")]
    assert "needs you on Colab" in notes[0].title and "1 needs you" in w.summary()
    notes = w.update(_ov(("a", "completed"), ("b", "completed"), ("c", "failed_user")))
    assert [(n.kind, n.job_id) for n in notes] == [("completed", "b"), ("failed", "c")]
    assert "30 min" in notes[0].message and "boom" in notes[1].message
    assert w.update(_ov(("a", "completed"), ("b", "completed"), ("c", "failed_user"))) == []


# ---------------------------------------------------------------- autostart and icon


class FakeReg:
    HKEY_CURRENT_USER = "HKCU"
    REG_SZ = 1

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def OpenKey(self, root: str, path: str) -> FakeReg:
        return self

    CreateKey = OpenKey

    def __enter__(self) -> FakeReg:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def QueryValueEx(self, key: Any, name: str) -> tuple[str, int]:
        if name not in self.values:
            raise OSError("missing")
        return self.values[name], self.REG_SZ

    def SetValueEx(self, key: Any, name: str, _r: int, _t: int, value: str) -> None:
        self.values[name] = value

    def DeleteValue(self, key: Any, name: str) -> None:
        if name not in self.values:
            raise OSError("missing")
        del self.values[name]


def test_autostart_uses_the_users_run_key_only() -> None:
    reg = FakeReg()
    assert not autostart.is_enabled(reg)
    autostart.set_enabled(True, '"C:/GPU Relay/GPU Relay.exe"', reg=reg)
    assert autostart.is_enabled(reg) and reg.values == {"GPU Relay": '"C:/GPU Relay/GPU Relay.exe"'}
    autostart.set_enabled(False, reg=reg)
    autostart.set_enabled(False, reg=reg)  # already off: fine
    assert not autostart.is_enabled(reg)
    assert "gpurelay.desktop" in autostart.launch_command()


def test_icon_has_every_size_and_a_badge(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from gpurelay.desktop.icon import make_icon, save_ico

    plain, badged = make_icon(64), make_icon(64, badge=True)
    assert plain.size == (64, 64) and plain.mode == "RGBA"
    assert plain.getpixel((32, 10))[2] > 150  # blue square
    assert badged.getpixel((50, 13)) != plain.getpixel((50, 13))  # the orange dot
    ico = save_ico(tmp_path / "gpurelay.ico")
    assert ico.read_bytes()[:4] == b"\x00\x00\x01\x00"


# ---------------------------------------------------------------- setup wizard API


class Hooks:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.events: list[str] = []
        self.auto = False

    def controller_state(self) -> dict[str, Any]:
        return {"mode": DesktopSettings.load(self.home).mode, "running": False,
                "enabled": True, "lease_holder": None, "elsewhere": False}  # fmt: skip

    def start_controller(self) -> None:
        self.events.append("start")

    def stop_controller(self) -> None:
        self.events.append("stop")

    def reconfigured(self) -> None:
        self.events.append("reconfigured")

    def connect_server(self, url: str) -> None:
        self.events.append(f"server {url}")

    def autostart(self) -> bool:
        return self.auto

    def set_autostart(self, enabled: bool) -> None:
        self.auto = enabled

    def packaged(self) -> bool:
        return self.is_packaged

    is_packaged = False
    menu = False

    def start_menu(self) -> bool:
        return self.menu

    def set_start_menu(self, enabled: bool) -> None:
        self.menu = enabled

    def update_info(self) -> dict[str, Any]:
        return {"current": "1.0.0", "latest": "1.1.0", "available": True, "url": "u"}

    def remove_from_windows(self) -> list[str]:
        self.events.append("removed")
        return ["Start menu entry"]


class FakeHfApi:
    repos: ClassVar[dict[str, bool]] = {}  # repo id -> private

    def __init__(self, token: str | None) -> None:
        self.token = token

    def whoami(self, token: str | None = None) -> dict[str, str]:
        if (token or self.token) != HF:
            raise RuntimeError("401 invalid token")
        return {"name": "me"}

    def repo_info(self, repo_id: str, repo_type: str = "", token: str | None = None) -> Any:
        if repo_id not in self.repos:
            raise RuntimeError("404")

        class Info:
            private = self.repos[repo_id]

        return Info()

    def create_repo(self, repo_id: str, repo_type: str = "", private: bool = False,
                    exist_ok: bool = False) -> None:  # fmt: skip
        self.repos[repo_id] = private


class FakeKaggle:
    username = "kuser"

    def __init__(self) -> None:
        self.created: list[str] = []

    def dataset_exists(self, ref: str) -> bool:
        return bool(self.created)

    def create_dataset(self, folder: Path) -> None:
        self.created.append((folder / SECRETS_FILE).read_text())


@pytest.fixture
def wizard(tmp_path: Path) -> tuple[TestClient, Hooks, dict[str, str], FakeKaggle]:
    FakeHfApi.repos = {}
    hooks = Hooks(tmp_path / "home")
    keyring: dict[str, str] = {}
    kaggle = FakeKaggle()
    token_file = tmp_path / "kaggle" / "access_token"
    deps = SetupDeps(
        hf_api=FakeHfApi,
        kaggle_client=lambda: kaggle,
        check_server=lambda url: None if "good" in url else (_ for _ in ()).throw(
            GpuRelayError(f"could not reach {url}")),
        kaggle_token_path=lambda: token_file,
        kaggle_present=token_file.is_file,
        get_secret=lambda name, env: keyring.get(name),
        set_secret=lambda name, value: keyring.__setitem__(name, value) is None,
        modal_config_path=lambda: tmp_path / ".modal.toml",
        modal_present=lambda: (tmp_path / ".modal.toml").is_file(),
    )  # fmt: skip
    app = create_setup_app(hooks, launch_key=KEY, deps=deps)
    client = TestClient(app, follow_redirects=False)
    assert client.get(f"/?k={KEY}").status_code == 303  # the window's entry URL
    return client, hooks, keyring, kaggle


def test_launch_key_guards_everything(tmp_path: Path) -> None:
    hooks = Hooks(tmp_path)
    client = TestClient(create_setup_app(hooks, launch_key=KEY), follow_redirects=False)
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 403
    assert client.get("/api/desktop/status").status_code == 403
    assert client.get("/?k=wrong").status_code == 403
    r = client.get(f"/?k={KEY}")
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.cookies.get(LAUNCH_COOKIE) == KEY
    assert client.get("/").status_code == 200
    assert client.get("/api/me").json()["configured"] is False


def test_wizard_hugging_face_kaggle_colab_and_finish(wizard: Any) -> None:
    client, hooks, keyring, kaggle = wizard
    home = hooks.home
    status = client.get("/api/desktop/status").json()
    assert status["configured"] is False and status["storage"]["token"] is False

    assert client.post("/api/setup/hf", json={"token": HF}).status_code == 403  # header needed
    bad = client.post("/api/setup/hf", json={"token": "hf_wrong"}, headers=ACT)
    assert bad.status_code == 400 and "did not accept" in bad.json()["detail"]
    assert client.post("/api/setup/hf", json={"token": KGAT}, headers=ACT).status_code == 400
    r = client.post("/api/setup/hf", json={"token": f"  {HF}  "}, headers=ACT)
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "repo": "me/gpurelay-store", "user": "me", "created": True,
                        "keyring": True}  # fmt: skip
    assert FakeHfApi.repos == {"me/gpurelay-store": True}  # created private
    assert keyring == {"hf_token": HF} or list(keyring.values()) == [HF]
    config_text = (home / CONFIG_FILENAME).read_text()
    assert HF not in config_text and HF not in r.text

    nokey = client.post("/api/setup/kaggle", json={}, headers=ACT)
    assert nokey.status_code == 400 and "Kaggle API token" in nokey.json()["detail"]
    r = client.post("/api/setup/kaggle", headers=ACT, json={
        "kaggle_token": KGAT, "accelerator": "NvidiaTeslaP100", "weekly_hours": 25})  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "username": "kuser", "created": True}
    cfg = load_config(home / CONFIG_FILENAME)
    assert cfg.backends.kaggle.enabled and cfg.backends.kaggle.username == "kuser"
    assert cfg.backends.kaggle.accelerator == "NvidiaTeslaP100"
    assert cfg.backends.kaggle.weekly_gpu_hours == 25
    assert HF in kaggle.created[0]  # the session token went to the private Kaggle dataset only
    assert KGAT not in (home / CONFIG_FILENAME).read_text() and KGAT not in r.text

    assert client.post("/api/setup/colab", json={"enabled": True}, headers=ACT).status_code == 200
    assert load_config(home / CONFIG_FILENAME).backends.colab_manual.enabled

    r = client.post("/api/setup/finish", json={"autostart": True}, headers=ACT)
    assert r.status_code == 200 and hooks.auto is True
    assert hooks.events == ["reconfigured"]
    assert DesktopSettings.load(home).mode == LOCAL
    status = client.get("/api/desktop/status").json()
    assert status["configured"] and status["kaggle"]["credentials"] and status["autostart"]


def test_wizard_local_storage_and_bad_inputs(wizard: Any) -> None:
    client, hooks, _, _ = wizard
    early = client.post("/api/setup/finish", json={}, headers=ACT)
    assert early.status_code == 400
    assert client.post("/api/setup/kaggle", json={}, headers=ACT).status_code == 400  # no storage
    assert client.post("/api/setup/local-storage", headers=ACT).status_code == 200
    cfg = load_config(hooks.home / CONFIG_FILENAME)
    assert cfg.storage.kind == "localfs" and cfg.backends.local.enabled
    for body in ({"token": "hf_a b"}, {"token": "x" * 301}, {"token": HF, "extra": 1}):
        assert client.post("/api/setup/hf", json=body, headers=ACT).status_code in (400, 422)
    acc = client.post("/api/setup/kaggle", json={"accelerator": "H100"}, headers=ACT)
    assert acc.status_code == 400


def test_wizard_connects_to_a_server(wizard: Any) -> None:
    client, hooks, _, _ = wizard
    r = client.post("/api/setup/server", json={"url": "good-relay.onrender.com"}, headers=ACT)
    assert r.json() == {"ok": True, "url": "https://good-relay.onrender.com"}
    assert hooks.events == ["server https://good-relay.onrender.com"]
    down = client.post("/api/setup/server", json={"url": "down.onrender.com"}, headers=ACT)
    assert down.status_code == 400 and "could not reach" in down.json()["detail"]
    plain = client.post("/api/setup/server", json={"url": "http://x.com"}, headers=ACT)
    assert plain.status_code == 400 and "https" in plain.json()["detail"]


def test_controller_buttons(wizard: Any) -> None:
    client, hooks, _, _ = wizard
    assert client.post("/api/controller/stop", headers=ACT).json()["ok"]
    assert client.post("/api/controller/start", headers=ACT).json()["ok"]
    assert client.post("/api/controller/reboot", headers=ACT).status_code == 404
    assert client.post("/api/controller/stop").status_code == 403
    assert hooks.events == ["stop", "start"]


# ---------------------------------------------------------------- the app itself (no window)


def test_desktop_app_switches_from_setup_to_the_full_app(tmp_path: Path) -> None:
    from gpurelay.desktop.app import DesktopApp

    home = tmp_path / "home"
    app = DesktopApp(home)
    assert app.ctx is None and not app.supervisor.enabled  # nothing set up yet
    client = TestClient(app.switch, base_url="http://127.0.0.1", follow_redirects=False)
    client.get(f"/?k={app.key}")
    assert client.get("/api/me").json()["configured"] is False
    assert app.entry_url().startswith(f"http://127.0.0.1:{app.port}/?k=")

    cfg = Config(storage=StorageSettings(kind="localfs", localfs_root=tmp_path / "store"))
    cfg.backends.kaggle.enabled = False
    save_config(cfg, home / CONFIG_FILENAME)
    DesktopSettings(mode=LOCAL).save(home)
    app._reload()
    assert app.ctx is not None and app.supervisor.enabled
    me = client.get("/api/me").json()
    assert me["configured"] is True and me["desktop"] is True
    assert client.get("/api/overview").status_code == 200
    state = app.controller_state()
    assert state["mode"] == LOCAL and state["elsewhere"] is False

    app.connect_server("https://good-relay.onrender.com")
    app._reload()
    assert app.entry_url("#/submit") == "https://good-relay.onrender.com/#/submit"
    assert not app.supervisor.enabled  # the server runs the controller


def test_create_app_refuses_desktop_mode_behind_a_login(tmp_path: Path) -> None:
    from gpurelay.dashboard.app import create_app
    from gpurelay.dashboard.auth import Auth

    with pytest.raises(ValueError, match="launch key"):
        create_app(None, desktop=Hooks(tmp_path), auth=Auth("owner-password-123"),  # type: ignore[arg-type]
                   launch_key=KEY)  # fmt: skip


def test_wizard_adds_modal_only_with_terms_and_a_token(wizard: Any) -> None:
    client, hooks, _, _ = wizard
    nostore = client.post("/api/setup/modal", json={"terms_reviewed": True}, headers=ACT)
    assert nostore.status_code == 400 and "Hugging Face storage" in nostore.json()["detail"]
    assert client.post("/api/setup/hf", json={"token": HF}, headers=ACT).status_code == 200
    noterms = client.post("/api/setup/modal", json={"token_id": "ak-1", "token_secret": "as-2"},
                          headers=ACT)  # fmt: skip
    assert noterms.status_code == 400 and "terms" in noterms.json()["detail"]
    half = client.post("/api/setup/modal", json={"terms_reviewed": True, "token_id": "ak-1"},
                       headers=ACT)  # fmt: skip
    assert half.status_code == 400 and "both" in half.json()["detail"]
    body = {"terms_reviewed": True, "token_id": "ak-1", "token_secret": "as-2", "gpu": "L4"}
    ok = client.post("/api/setup/modal", headers=ACT, json=body)
    assert ok.status_code == 200, ok.text
    modal_file = hooks.home.parent / ".modal.toml"
    assert 'token_secret = "as-2"' in modal_file.read_text()
    cfg = load_config(hooks.home / CONFIG_FILENAME)
    assert cfg.backends.modal.enabled and cfg.backends.modal.terms_reviewed
    assert cfg.backends.modal.gpu == "L4"
    assert "as-2" not in (hooks.home / CONFIG_FILENAME).read_text() and "as-2" not in ok.text
    again = client.post("/api/setup/modal", headers=ACT, json={
        "terms_reviewed": True, "token_id": "ak-9", "token_secret": "as-9"})  # fmt: skip
    assert again.status_code == 400 and "modal token set" in again.json()["detail"]
    assert 'token_secret = "as-2"' in modal_file.read_text()  # never overwritten
    status = client.get("/api/desktop/status").json()
    assert status["modal"]["enabled"] and status["modal"]["credentials"]


# ---------------------------------------------------------------- D4: Windows integration


def test_update_check_compares_versions_and_survives_being_offline() -> None:
    from gpurelay.desktop.integration import check_for_update

    def release(tag: str) -> Any:
        return lambda url: (
            f'{{"tag_name": "{tag}", "html_url": "https://github.com/x/releases/{tag}"}}'
        ).encode()

    newer = check_for_update("1.0.0", fetch=release("v1.1.0"))
    assert newer.available and newer.latest == "1.1.0" and newer.url and "1.1.0" in newer.url
    assert not check_for_update("1.0.0", fetch=release("v1.0.0")).available
    assert not check_for_update("1.2.0", fetch=release("1.1.9")).available
    assert check_for_update("1.0.0", fetch=release("1.0.10")).available  # numeric, not text

    def offline(url: str) -> bytes:
        raise OSError("no network")

    info = check_for_update("1.0.0", fetch=offline)
    assert info.latest is None and not info.available


@pytest.mark.skipif(sys.platform != "win32", reason="Start menu entries are Windows-only")
def test_start_menu_shortcut_and_removal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gpurelay.desktop import integration

    monkeypatch.setenv("APPDATA", str(tmp_path))
    scripts: list[str] = []

    def fake_run(cmd: list[str], **kw: Any) -> None:
        scripts.append(cmd[-1])
        integration.shortcut_path().write_text("lnk")

    link = integration.create_shortcut("C:/Apps/O'Neil/GPU Relay.exe", run=fake_run)
    assert link == tmp_path / "Microsoft/Windows/Start Menu/Programs/GPU Relay.lnk"
    assert "'C:/Apps/O''Neil/GPU Relay.exe'" in scripts[0]  # quoted for PowerShell
    assert integration.has_shortcut()
    reg = FakeReg()
    monkeypatch.setattr(autostart, "_winreg", lambda: reg)
    autostart.set_enabled(True, "x", reg=reg)
    assert integration.remove_from_windows() == ["Start menu entry", "start with Windows"]
    assert not integration.has_shortcut() and not autostart.is_enabled(reg)
    assert integration.remove_from_windows() == []


def test_start_menu_and_remove_endpoints(wizard: Any) -> None:
    client, hooks, _, _ = wizard
    refused = client.post("/api/desktop/start-menu", json={"enabled": True}, headers=ACT)
    assert refused.status_code == 400 and "packaged" in refused.json()["detail"]
    hooks.is_packaged = True
    r = client.post("/api/desktop/start-menu", json={"enabled": True}, headers=ACT)
    assert r.json() == {"ok": True, "start_menu": True}
    status = client.get("/api/desktop/status").json()
    assert status["packaged"] and status["start_menu"] and status["update"]["available"]
    assert client.post("/api/desktop/remove", headers=ACT).json()["removed"] == ["Start menu entry"]
    assert client.post("/api/desktop/remove").status_code == 403  # header required
