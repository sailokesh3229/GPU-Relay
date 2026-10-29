"""Windows integration for the packaged app (plan D4): a Start menu entry, the update check,
and removing what the app added to Windows before you delete its folder.

The app is portable (unzip and run), so there is no installer: these do an installer's jobs.
None of them touches your jobs, storage, tokens or the GPU Relay home folder.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gpurelay.desktop import autostart

log = logging.getLogger(__name__)

REPO = "sailokesh3229/GPU-Relay"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"
SHORTCUT_NAME = "GPU Relay.lnk"


def start_menu_dir() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def shortcut_path() -> Path:
    return start_menu_dir() / SHORTCUT_NAME


def has_shortcut() -> bool:
    return shortcut_path().exists()


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def create_shortcut(target: str | None = None, run: Callable[..., object] = subprocess.run) -> Path:
    """Add GPU Relay to the Start menu (current user only), pointing at this executable."""
    if os.name != "nt":  # (os.name: mypy checks the rest on every platform)
        raise OSError("Start menu entries exist only on Windows")
    exe = target or sys.executable
    link = shortcut_path()
    link.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _ps_quote(str(link)) + ");"
        "$s.TargetPath = " + _ps_quote(exe) + ";"
        "$s.WorkingDirectory = " + _ps_quote(str(Path(exe).parent)) + ";"
        "$s.Description = 'GPU Relay: relay ML jobs across free GPU tiers';"
        "$s.Save()"
    )
    run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, timeout=60, capture_output=True)  # fmt: skip
    return link


def remove_shortcut() -> None:
    shortcut_path().unlink(missing_ok=True)


def remove_from_windows() -> list[str]:
    """Undo what the app added to Windows: the Start menu entry and start-with-Windows. Your
    jobs, storage and tokens stay. Returns what was removed."""
    removed = []
    if has_shortcut():
        remove_shortcut()
        removed.append("Start menu entry")
    if autostart.is_enabled():
        autostart.set_enabled(False)
        removed.append("start with Windows")
    return removed


def _version_tuple(text: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", text.split("-")[0].split("+")[0])
    return tuple(int(n) for n in numbers[:3]) or (0,)


@dataclass(frozen=True)
class UpdateInfo:
    current: str
    latest: str | None
    url: str | None

    @property
    def available(self) -> bool:
        return self.latest is not None and _version_tuple(self.latest) > _version_tuple(
            self.current
        )


def check_for_update(current: str, fetch: Callable[[str], bytes] | None = None) -> UpdateInfo:
    """The newest release on GitHub (no account needed). Never raises: offline means "no
    update known"."""

    def default_fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                       "User-Agent": "gpurelay"})  # fmt: skip
        with urllib.request.urlopen(request, timeout=15) as response:
            return bytes(response.read(65536))

    try:
        data = json.loads((fetch or default_fetch)(RELEASES_API))
        tag = str(data.get("tag_name") or "").lstrip("v") or None
        url = data.get("html_url")
        return UpdateInfo(current, tag, str(url) if url else None)
    except Exception as exc:
        log.debug("update check failed: %s", exc)
        return UpdateInfo(current, None, None)
