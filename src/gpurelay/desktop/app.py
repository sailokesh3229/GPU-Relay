"""The GPU Relay desktop app: native window (pywebview), tray icon (pystray), notifications.

Pieces, all in one process except the controller:

* a local web server on 127.0.0.1 (the same web app as ``gpurelay serve``, plus the setup
  wizard), behind a per-launch key;
* in "this computer" mode, the controller as a supervised child process (``gpurelay up``),
  started and stopped from the window or the tray, stopped gracefully on quit;
* in "my server" mode, the window shows the server's web app instead;
* a poller that turns job changes into tray notifications and keeps the tooltip current.

Closing the window keeps the app in the tray; "Quit" in the tray menu ends it. GPU sessions
already running on Kaggle continue either way.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from gpurelay import __version__
from gpurelay.config import gpurelay_home, load_config
from gpurelay.constants import CONFIG_FILENAME
from gpurelay.desktop import autostart
from gpurelay.desktop.settings import LOCAL, SERVER, DesktopSettings, InstanceLock
from gpurelay.desktop.watcher import JobWatcher
from gpurelay.hosted import Supervisor, controller_command, spawn_controller

log = logging.getLogger("gpurelay.desktop")


def gpurelay_version() -> str:
    return __version__


POLL_S = 30.0
WINDOW_SIZE = (1280, 820)
MIN_SIZE = (960, 560)


class Switch:
    """An ASGI app that forwards to whichever app is current (swapped after setup)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class DesktopApp:
    """Implements :class:`gpurelay.dashboard.desktop_api.DesktopHooks` for the web app."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = DesktopSettings.load(home)
        self.key = secrets.token_urlsafe(24)
        self.port = _free_port()
        self.ctx: Any = None
        self.supervisor = Supervisor(
            controller_command(False),
            backup=lambda: False,  # the local database is the source of truth: no backup
            backup_s=float("inf"),
            spawn=lambda cmd: spawn_controller(cmd, home / "logs" / "controller.log"),
            home=home,
        )
        self.switch = Switch(self._build_app())
        self.supervisor.enabled = self._controller_wanted()
        self.watcher = JobWatcher()
        self.window: Any = None
        self.tray: Any = None
        self.server: Any = None
        self.quitting = False
        self._hint_shown = False
        self._lock = threading.Lock()
        self._lease: tuple[float, str | None] = (0.0, None)
        self.update: Any = None  # UpdateInfo, filled in the background at start

    # ------------------------------------------------------------------ web app

    def _configured(self) -> bool:
        return self.settings.mode == LOCAL and (self.home / CONFIG_FILENAME).exists()

    def _controller_wanted(self) -> bool:
        return self._configured() and self.ctx is not None

    def _build_app(self) -> Any:
        from gpurelay.dashboard.app import LOCAL_HOSTS, create_app
        from gpurelay.dashboard.desktop_api import create_setup_app

        self.ctx = None
        if not self._configured():
            return create_setup_app(self, launch_key=self.key)
        try:
            from gpurelay.controller.service import open_context

            self.ctx = open_context(load_config(self.home / CONFIG_FILENAME), self.home)
        except Exception as exc:  # broken config or missing token: the wizard shows why
            log.warning("cannot open GPU Relay: %s", exc)
            return create_setup_app(self, launch_key=self.key, error=str(exc))
        return create_app(self.ctx, allowed_hosts=LOCAL_HOSTS, desktop=self, launch_key=self.key)

    def local_url(self, path: str = "/") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def entry_url(self, fragment: str = "") -> str:
        if self.settings.mode == SERVER and self.settings.server_url:
            return f"{self.settings.server_url}/{fragment}"
        return self.local_url(f"/?k={self.key}{fragment}")

    # ------------------------------------------------------------------ DesktopHooks

    def controller_state(self) -> dict[str, Any]:
        holder = self._lease_holder()
        ours = holder is None or holder.startswith(f"{socket.gethostname()}:")
        return {
            "mode": self.settings.mode,
            "running": self.supervisor.running,
            "enabled": self.supervisor.enabled,
            "lease_holder": holder,
            "elsewhere": not ours,  # another machine (e.g. your server) runs the controller
        }

    def _lease_holder(self) -> str | None:
        """Who holds the store's controller lease (read at most once a minute)."""
        checked, holder = self._lease
        if self.ctx is None or time.monotonic() - checked < 60:
            return holder
        try:
            from gpurelay.controller.lease import LEASE_KEY

            raw = self.ctx.storage.get_bytes(LEASE_KEY)
            holder = str(json.loads(raw)["holder"]) if raw else None
        except Exception as exc:
            log.debug("lease read failed: %s", exc)
        self._lease = (time.monotonic(), holder)
        return holder

    def start_controller(self) -> None:
        if self._controller_wanted():
            self.supervisor.resume()
            self.supervisor.step()

    def stop_controller(self) -> None:
        self.supervisor.halt()

    def reconfigured(self) -> None:
        """Settings or tokens changed: rebuild the web app, restart the controller."""
        threading.Thread(target=self._reload, name="gpurelay-reload", daemon=True).start()

    def _reload(self) -> None:
        with self._lock:
            time.sleep(0.5)  # let the request that triggered this finish
            self.supervisor.halt()
            self.settings = DesktopSettings.load(self.home)
            self._lease = (0.0, None)
            self.switch.app = self._build_app()
            if self._controller_wanted():
                self.supervisor.resume()
            self.watcher = JobWatcher()
            if self.window is not None:
                self.window.load_url(self.entry_url())

    def connect_server(self, url: str) -> None:
        self.settings = DesktopSettings(mode=SERVER, server_url=url)
        self.settings.save(self.home)
        self.reconfigured()

    def autostart(self) -> bool:
        return autostart.is_enabled()

    def set_autostart(self, enabled: bool) -> None:
        autostart.set_enabled(enabled)

    def packaged(self) -> bool:
        return bool(getattr(sys, "frozen", False))

    def start_menu(self) -> bool:
        from gpurelay.desktop import integration

        return integration.has_shortcut()

    def set_start_menu(self, enabled: bool) -> None:
        from gpurelay.desktop import integration

        if enabled:
            integration.create_shortcut()
        else:
            integration.remove_shortcut()

    def update_info(self) -> dict[str, Any]:
        info = self.update
        if info is None:
            return {"current": gpurelay_version(), "latest": None, "available": False, "url": None}
        return {"current": info.current, "latest": info.latest, "available": info.available,
                "url": info.url}  # fmt: skip

    def remove_from_windows(self) -> list[str]:
        from gpurelay.desktop import integration

        return integration.remove_from_windows()

    def _check_update(self) -> None:
        from gpurelay.desktop.integration import check_for_update

        self.update = check_for_update(gpurelay_version())
        if self.update.available:
            self.notify(
                f"GPU Relay {self.update.latest} is available",
                "Download it from the Releases page on GitHub (Settings shows the link).",
            )

    # ------------------------------------------------------------------ jobs and tray

    def _overview(self) -> dict[str, Any] | None:
        if self.settings.mode == SERVER and self.settings.server_url:
            url, headers = f"{self.settings.server_url}/api/overview", {}
        elif self.ctx is not None:
            from gpurelay.dashboard.desktop_api import LAUNCH_COOKIE

            url = self.local_url("/api/overview")
            headers = {"Cookie": f"{LAUNCH_COOKIE}={self.key}"}
        else:
            return None
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                data: dict[str, Any] = json.loads(response.read())
                return data
        except Exception as exc:  # offline, server asleep: try again next time
            log.debug("overview failed: %s", exc)
            return None

    def poll_once(self) -> None:
        data = self._overview()
        if data is None:
            return
        for note in self.watcher.update(data):
            self.notify(note.title, note.message)
        if self.tray is not None:
            self.tray.title = self.watcher.summary()
            from gpurelay.desktop.icon import make_icon

            self.tray.icon = make_icon(64, badge=self.watcher.needs_you > 0)

    def notify(self, title: str, message: str) -> None:
        log.info("notify: %s: %s", title, message)
        if self.tray is not None:
            try:
                self.tray.notify(message[:250], title[:60])
            except Exception as exc:
                log.debug("tray notification failed: %s", exc)

    def _poll_loop(self, lock: InstanceLock) -> None:
        last = 0.0
        while not self.quitting:
            if lock.show_requested():
                self.show()
            if time.monotonic() - last >= POLL_S:
                last = time.monotonic()
                self.poll_once()
            time.sleep(1.0)

    def show(self, fragment: str = "") -> None:
        if self.window is None:
            return
        if fragment:
            self.window.load_url(self.entry_url(fragment))
        self.window.show()
        self.window.restore()

    def pause_all(self) -> None:
        if self.ctx is None:
            return
        from gpurelay.controller.service import request

        paused = 0
        for job in self.ctx.db.list_jobs():
            if not job.state.is_terminal and job.state.value != "paused":
                try:
                    request(self.ctx, job.id, "pause")
                    paused += 1
                except Exception as exc:
                    log.warning("pause %s failed: %s", job.id, exc)
        self.notify("Pausing jobs", f"{paused} job(s) pause at their next safe point.")

    def quit(self) -> None:
        self.quitting = True
        log.info("quitting")
        if self.tray is not None:
            self.tray.stop()
        self.supervisor.stop()  # graceful: the lease is released; GPU sessions continue
        if self.server is not None:
            self.server.should_exit = True
        if self.window is not None:
            self.window.destroy()

    def _toggle_controller(self) -> None:
        action = self.stop_controller if self.supervisor.enabled else self.start_controller
        threading.Thread(target=action, name="gpurelay-toggle", daemon=True).start()

    def _tray_menu(self) -> Any:
        import pystray

        local = lambda item: self.settings.mode == LOCAL  # noqa: E731
        return pystray.Menu(
            pystray.MenuItem("Open GPU Relay", lambda: self.show(), default=True),
            pystray.MenuItem("Submit job…", lambda: self.show("#/submit")),
            pystray.MenuItem("GPU budget", lambda: self.show("#/budget")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: "Stop controller" if self.supervisor.enabled else "Start controller",
                lambda: self._toggle_controller(),
                visible=local,
            ),
            pystray.MenuItem("Pause all jobs", lambda: self.pause_all(), visible=local),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit GPU Relay", lambda: self.quit()),
        )

    def _on_closing(self) -> bool:
        if self.quitting:
            return True
        self.window.hide()
        if not self._hint_shown:
            self._hint_shown = True
            self.notify(
                "GPU Relay is still running",
                "It keeps relaying your jobs. Use the tray icon to open it or quit.",
            )
        return False  # keep the app running in the tray

    # ------------------------------------------------------------------ run

    def run(self, lock: InstanceLock) -> None:
        import pystray
        import uvicorn
        import webview

        from gpurelay.desktop.icon import make_icon

        config = uvicorn.Config(self.switch, host="127.0.0.1", port=self.port,
                                log_level="warning", access_log=False)  # fmt: skip
        self.server = uvicorn.Server(config)
        threading.Thread(target=self.server.run, name="gpurelay-web", daemon=True).start()
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.05)
        log.info("local web app on 127.0.0.1:%d (mode %s)", self.port, self.settings.mode)
        if os.environ.get("GPURELAY_DESKTOP_DEV") == "1":  # testing: open it in a browser too
            log.info("dev entry URL: %s", self.local_url(f"/?k={self.key}"))
        threading.Thread(target=self.supervisor.run, name="gpurelay-supervisor",
                         daemon=True).start()  # fmt: skip

        self.tray = pystray.Icon("gpurelay", make_icon(64), "GPU Relay", self._tray_menu())
        self.tray.run_detached()
        threading.Thread(target=self._check_update, name="gpurelay-update", daemon=True).start()
        threading.Thread(target=self._poll_loop, args=(lock,), name="gpurelay-poll",
                         daemon=True).start()  # fmt: skip

        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
        width, height = WINDOW_SIZE
        try:  # fit small laptop screens (the taskbar must not hide the bottom of the window)
            screen = webview.screens[0]
            width, height = min(width, screen.width - 60), min(height, screen.height - 110)
        except Exception as exc:
            log.debug("screen size unknown: %s", exc)
        self.window = webview.create_window(
            "GPU Relay", self.entry_url(), width=max(width, MIN_SIZE[0]),
            height=max(height, 560),
            min_size=MIN_SIZE, background_color="#F3F4F6",
        )  # fmt: skip
        self.window.events.closing += self._on_closing
        storage = self.home / "webview"  # remembers the server login between launches
        dev = os.environ.get("GPURELAY_DESKTOP_DEV") == "1"
        webview.start(private_mode=False, storage_path=str(storage), debug=dev)
        if not self.quitting:  # the window was destroyed some other way
            self.quit()


def _setup_logging(home: Path) -> None:
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        handlers=[logging.FileHandler(logs / "desktop.log", encoding="utf-8")],
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "pywebview", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if os.environ.get("GPURELAY_DESKTOP_DEV") == "1":
        logging.getLogger("pywebview").setLevel(logging.DEBUG)


def main(argv: list[str] | None = None) -> None:
    """Start the desktop app. In the packaged executable, arguments run CLI commands
    instead (that is how the app starts its controller: ``GPU Relay.exe up``)."""
    args = sys.argv[1:] if argv is None else argv
    if os.environ.get("GPURELAY_DESKTOP_DEV") == "1":
        _setup_logging(gpurelay_home())
        log.info("process %d started with %r (frozen=%s)", os.getpid(), sys.argv,
                 getattr(sys, "frozen", False))  # fmt: skip
    if args and getattr(sys, "frozen", False):
        from gpurelay.cli import app

        app(args)
        return
    home = gpurelay_home()
    _setup_logging(home)
    lock = InstanceLock(home)
    if not lock.acquire():
        lock.ask_running_app_to_show()  # already running: bring its window forward
        return
    try:
        log.info("starting GPU Relay desktop (home %s)", home)
        DesktopApp(home).run(lock)
    except Exception:
        log.exception("the desktop app failed")
        raise
    finally:
        lock.release()


if __name__ == "__main__":  # pragma: no cover
    main()
