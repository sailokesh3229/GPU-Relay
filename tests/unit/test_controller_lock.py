"""Only one controller may run per GPU Relay home."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpurelay.controller.lock import LOCK_FILENAME, controller_lock
from gpurelay.errors import ControllerRunningError

HOLD = """
import sys, time
from pathlib import Path
from gpurelay.controller.lock import controller_lock
with controller_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    time.sleep(60)
"""


def test_lock_is_released_after_use(tmp_path: Path) -> None:
    with controller_lock(tmp_path):
        assert (tmp_path / LOCK_FILENAME).read_text() == str(os.getpid())
    with controller_lock(tmp_path):
        pass


def test_second_controller_is_refused_until_the_first_dies(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(tmp_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        with (
            pytest.raises(ControllerRunningError, match="already running"),
            controller_lock(tmp_path),
        ):
            pass
    finally:
        holder.kill()  # a killed controller must not leave a stale lock
        holder.wait()
    # Windows: the venv launcher's child process exits a moment after the launcher.
    deadline = time.monotonic() + 10
    while True:
        try:
            with controller_lock(tmp_path):
                return
        except ControllerRunningError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def test_controller_running_probe(tmp_path: Path) -> None:
    from gpurelay.controller.lock import controller_running

    assert not controller_running(tmp_path)
    with controller_lock(tmp_path):
        pass
    assert not controller_running(tmp_path)  # lock file exists now, still free
    with controller_lock(tmp_path):  # the probe left it free
        pass
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(tmp_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        assert controller_running(tmp_path)
    finally:
        holder.kill()
        holder.wait()
