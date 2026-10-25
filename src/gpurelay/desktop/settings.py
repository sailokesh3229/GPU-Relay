"""The desktop app's own settings (``desktop.json`` in the GPU Relay home, no secrets) and its
single-instance lock."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from gpurelay.controller.lock import _try_lock
from gpurelay.errors import ConfigError

SETTINGS_FILENAME = "desktop.json"
LOCK_FILENAME = "desktop.lock"
SHOW_FILENAME = "desktop.show"  # a second launch asks the running app to show its window
LOCAL = "local"
SERVER = "server"


@dataclass
class DesktopSettings:
    mode: str | None = None  # LOCAL | SERVER | None (not chosen yet: the wizard asks)
    server_url: str | None = None

    @classmethod
    def load(cls, home: Path) -> DesktopSettings:
        try:
            data = json.loads((home / SETTINGS_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        mode = data.get("mode")
        url = data.get("server_url")
        return cls(
            mode=mode if mode in (LOCAL, SERVER) else None,
            server_url=url if isinstance(url, str) else None,
        )

    def save(self, home: Path) -> None:
        home.mkdir(parents=True, exist_ok=True)
        tmp = home / (SETTINGS_FILENAME + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, home / SETTINGS_FILENAME)


def normalize_server_url(url: str) -> str:
    """A server's base URL: https (http only for this computer), no path, query or fragment."""
    raw = url.strip()
    if raw and "://" not in raw:
        raw = "https://" + raw
    parts = urlsplit(raw)
    host = (parts.hostname or "").lower()
    local = host in ("localhost", "127.0.0.1", "::1")
    if parts.scheme not in ("https", "http") or not host:
        raise ConfigError("enter the server's address, e.g. https://gpu-relay.onrender.com")
    if parts.scheme == "http" and not local:
        raise ConfigError("use the server's https:// address (http is only for this computer)")
    if parts.username or parts.password:
        raise ConfigError("the server address must not contain a user name or password")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ConfigError("enter only the server's address, without a path")
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    return f"{parts.scheme}://{netloc}"


class InstanceLock:
    """Held while the desktop app runs; a second launch finds it taken."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self._fd: int | None = None

    def acquire(self) -> bool:
        self.home.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.home / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o600)
        if not _try_lock(fd):
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)  # closing the descriptor releases the OS lock
            self._fd = None

    def ask_running_app_to_show(self) -> None:
        (self.home / SHOW_FILENAME).write_text("show\n", encoding="utf-8")

    def show_requested(self) -> bool:
        path = self.home / SHOW_FILENAME
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        return False
