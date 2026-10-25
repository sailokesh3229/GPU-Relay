"""`gpurelay serve` (plan W1): state backup/restore through storage, server config, supervisor."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from gpurelay.config import Config, StorageSettings, load_config, save_config
from gpurelay.constants import CONFIG_FILENAME, STATE_DB_FILENAME
from gpurelay.controller.db import StateDB
from gpurelay.errors import ConfigError
from gpurelay.hosted import (
    ENV_HF_REPO,
    STATE_KEY,
    Supervisor,
    backup_state,
    controller_command,
    hosted_config,
    prepare_home,
    restore_state,
)
from gpurelay.models import Job, JobKind
from gpurelay.storage.localfs import LocalFSStorage

REPO = "me/gpurelay-store"


def _laptop_home(path: Path) -> Path:
    """A home as `gpurelay init` + `submit` leave it: config plus a database with a job."""
    cfg = Config(storage=StorageSettings(kind="hf_hub", hf_repo_id=REPO))
    save_config(cfg, path / CONFIG_FILENAME)
    StateDB(path / STATE_DB_FILENAME).put_job(Job(name="train", kind=JobKind.SCRIPT, entry="t.py"))
    return path


def _job_names(home: Path) -> list[str]:
    db = StateDB(home / STATE_DB_FILENAME)
    try:
        return [j.name for j in db.list_jobs()]
    finally:
        db.close()


def test_backup_then_restore_moves_the_job_database(tmp_path: Path) -> None:
    storage = LocalFSStorage(tmp_path / "store")
    laptop = _laptop_home(tmp_path / "laptop")
    assert backup_state(storage, laptop) and storage.exists(STATE_KEY)

    server = tmp_path / "server"
    assert restore_state(storage, server)
    assert _job_names(server) == ["train"]
    assert load_config(server / CONFIG_FILENAME).storage.hf_repo_id == REPO


def test_restore_never_overwrites_a_local_database(tmp_path: Path) -> None:
    storage = LocalFSStorage(tmp_path / "store")
    backup_state(storage, _laptop_home(tmp_path / "laptop"))
    server = tmp_path / "server"
    server.mkdir()
    newer = Job(name="newer", kind=JobKind.SCRIPT, entry="n.py")
    StateDB(server / STATE_DB_FILENAME).put_job(newer)
    assert not restore_state(storage, server)
    assert _job_names(server) == ["newer"]
    assert not restore_state(LocalFSStorage(tmp_path / "nothing"), tmp_path / "fresh")
    assert not backup_state(storage, tmp_path / "fresh")  # no database yet: nothing to save


def test_server_config_turns_off_local_jobs_and_desktop_notes() -> None:
    cfg = Config()
    out = hosted_config(cfg)
    assert out.backends.local.enabled is False and out.notify.desktop is False
    assert out.backends.kaggle == cfg.backends.kaggle and out.runtime == cfg.runtime


def test_prepare_home_restores_and_writes_server_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalFSStorage(tmp_path / "store")
    backup_state(storage, _laptop_home(tmp_path / "laptop"))
    opened: list[str] = []

    def factory(repo: str) -> LocalFSStorage:
        opened.append(repo)
        return storage

    server = tmp_path / "server"
    monkeypatch.setenv(ENV_HF_REPO, REPO)
    assert prepare_home(server, factory) is storage and opened == [REPO]
    cfg = load_config(server / CONFIG_FILENAME)
    assert cfg.backends.local.enabled is False and cfg.notify.desktop is False
    assert _job_names(server) == ["train"]

    monkeypatch.setenv(ENV_HF_REPO, "someone/else")
    with pytest.raises(ConfigError, match="another storage repo"):
        prepare_home(server, factory)


def test_prepare_home_explains_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_HF_REPO, raising=False)
    with pytest.raises(ConfigError, match=ENV_HF_REPO):
        prepare_home(tmp_path / "a", lambda repo: LocalFSStorage(tmp_path / "s"))
    monkeypatch.setenv(ENV_HF_REPO, REPO)
    with pytest.raises(ConfigError, match="no state backup"):
        prepare_home(tmp_path / "b", lambda repo: LocalFSStorage(tmp_path / "s"))


class FakeChild:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def send_signal(self, sig: int) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0

    def kill(self) -> None:
        self.returncode = -9


class Rig:
    def __init__(self, backup_ok: bool = True) -> None:
        self.t = 0.0
        self.children: list[FakeChild] = []
        self.backups = 0
        self.backup_ok = backup_ok
        self.sup = Supervisor(["gpurelay", "up"], self.backup, backup_s=600,
                              spawn=self.spawn, clock=lambda: self.t)  # type: ignore[arg-type]  # fmt: skip

    def spawn(self, cmd: Sequence[str]) -> Any:
        assert list(cmd) == ["gpurelay", "up"]
        self.children.append(FakeChild())
        return self.children[-1]

    def backup(self) -> bool:
        self.backups += 1
        if not self.backup_ok:
            raise OSError("hub unreachable")
        return True


def test_supervisor_restarts_a_dead_controller_with_backoff() -> None:
    rig = Rig()
    rig.sup.step()
    assert len(rig.children) == 1
    rig.children[0].returncode = 1
    rig.t = 5
    rig.sup.step()
    assert len(rig.children) == 1 and rig.backups == 1  # backed up, waiting 10 s
    rig.t = 16
    rig.sup.step()
    assert len(rig.children) == 2
    rig.children[1].returncode = 1
    rig.t = 17
    rig.sup.step()
    rig.t = 40
    rig.sup.step()
    assert len(rig.children) == 2  # second crash: waits 30 s
    rig.t = 48
    rig.sup.step()
    assert len(rig.children) == 3

    rig.t += 700  # healthy for a while: the backoff starts over
    rig.sup.step()
    rig.children[2].returncode = 0
    rig.t += 1
    rig.sup.step()
    rig.t += 11
    rig.sup.step()
    assert len(rig.children) == 4


def test_supervisor_backs_up_periodically_and_survives_backup_errors() -> None:
    rig = Rig(backup_ok=False)
    rig.sup.step()
    for minute in range(1, 31):
        rig.t = minute * 60
        rig.sup.step()
    assert rig.backups == 3  # every 10 minutes, each failure only logged
    assert len(rig.children) == 1


def test_supervisor_stop_ends_the_controller_and_saves() -> None:
    rig = Rig()
    rig.sup.step()
    child = rig.children[0]
    rig.sup.stop()
    assert child.terminated and rig.backups == 1
    rig.sup.step()  # after stop nothing starts again
    assert len(rig.children) == 1


def test_controller_command_runs_this_python() -> None:
    import sys

    assert controller_command(False) == [sys.executable, "-m", "gpurelay", "up"]
    assert controller_command(True)[-1] == "--takeover"


def test_supervisor_halts_with_a_stop_file_and_resumes(tmp_path: Path) -> None:
    from gpurelay.hosted import STOP_FILENAME

    rig = Rig()
    rig.sup.home = tmp_path
    rig.sup.step()
    child = rig.children[0]
    child.wait = lambda timeout=None: 0  # type: ignore[method-assign]  # exits when asked
    rig.sup.halt(timeout_s=1)
    assert (tmp_path / STOP_FILENAME).exists() and not child.terminated  # asked, not killed
    rig.t += 1000
    rig.sup.step()
    assert len(rig.children) == 1 and not rig.sup.running  # halted: nothing restarts
    rig.sup.resume()
    rig.sup.step()
    assert len(rig.children) == 2 and rig.sup.running


def test_stop_file_ends_a_sleeping_process_like_ctrl_c(tmp_path: Path) -> None:
    """Real process: `up` sleeps between ticks; the stop file must interrupt it at once."""
    import subprocess
    import sys
    import time

    from gpurelay.hosted import STOP_FILENAME

    code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from gpurelay.hosted import watch_stop_file\n"
        "watch_stop_file(Path(sys.argv[1]), poll_s=0.1)\n"
        "try:\n"
        "    time.sleep(60)\n"
        "except KeyboardInterrupt:\n"
        "    print('stopped gracefully'); sys.exit(0)\n"
        "sys.exit(3)\n"
    )
    (tmp_path / STOP_FILENAME).write_text("old request")  # leftovers are ignored
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                            stdout=subprocess.PIPE, text=True)  # fmt: skip
    time.sleep(1.5)
    assert proc.poll() is None  # the leftover did not stop it
    started = time.monotonic()
    (tmp_path / STOP_FILENAME).write_text("stop")
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 0 and "stopped gracefully" in out
    assert time.monotonic() - started < 5
    assert not (tmp_path / STOP_FILENAME).exists()
