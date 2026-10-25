"""Start GPU Relay when you sign in to Windows (the user's own Run key; opt-in only).

Elsewhere this is a no-op that reports "not supported".
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Sequence
from typing import Any

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "GPU Relay"


def supported() -> bool:
    return sys.platform == "win32"


def _winreg() -> Any:
    import winreg

    return winreg


def launch_command() -> str:
    """How Windows should start the app: the packaged executable, or pythonw for a source
    install (no console window)."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    exe = sys.executable
    pythonw = (
        exe[: -len("python.exe")] + "pythonw.exe" if exe.lower().endswith("python.exe") else exe
    )
    return f'"{pythonw}" -m gpurelay.desktop'


def is_enabled(reg: Any = None) -> bool:
    if not supported() and reg is None:
        return False
    reg = reg or _winreg()
    try:
        with reg.OpenKey(reg.HKEY_CURRENT_USER, RUN_KEY) as key:
            reg.QueryValueEx(key, VALUE_NAME)
        return True
    except OSError:
        return False


def set_enabled(enabled: bool, command: str | None = None, reg: Any = None) -> None:
    """Add or remove the Run entry for the current user only."""
    if not supported() and reg is None:
        return
    reg = reg or _winreg()
    with reg.CreateKey(reg.HKEY_CURRENT_USER, RUN_KEY) as key:
        if enabled:
            reg.SetValueEx(key, VALUE_NAME, 0, reg.REG_SZ, command or launch_command())
        else:
            with contextlib.suppress(OSError):
                reg.DeleteValue(key, VALUE_NAME)


__all__: Sequence[str] = ["is_enabled", "launch_command", "set_enabled", "supported"]
