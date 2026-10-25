"""``gpurelay serve``: the controller and the dashboard in one always-on server (plan W1).

Made for a free container host such as Render, whose disk is wiped on every restart (and,
on the free plan, every time the service sleeps). Everything durable already lives in the
storage repo except the job database, so this module keeps a copy of it there too
(``controller/state.tar.gz``, the same archive as ``export-state``): restored when the server
starts with an empty home, backed up every few minutes and on shutdown.

The controller runs as a child process (``gpurelay up``), exactly as on a laptop, and is
restarted when it exits; the dashboard runs in this process behind a login.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from gpurelay.config import Config, load_config, save_config
from gpurelay.constants import CONFIG_FILENAME, STATE_DB_FILENAME
from gpurelay.controller.migrate import export_state, import_state
from gpurelay.controller.service import open_storage_from_config
from gpurelay.errors import ConfigError
from gpurelay.storage.base import Storage

log = logging.getLogger(__name__)

STATE_KEY = "controller/state.tar.gz"
ENV_HF_REPO = "GPURELAY_HF_REPO"
ENV_BACKUP_S = "GPURELAY_BACKUP_S"
ENV_TAKEOVER = "GPURELAY_TAKEOVER"
ENV_MODAL = "GPURELAY_MODAL"
ENV_MODAL_GPU = "GPURELAY_MODAL_GPU"
ENV_MODAL_PAID_CAP = "GPURELAY_MODAL_PAID_CAP"  # USD a month after the free credit; default 0


def _usd(value: str | None) -> float:
    """A monthly limit from the environment; anything unclear means $0 (never pay)."""
    try:
        amount = float((value or "0").strip().lstrip("$"))
    except ValueError:
        return 0.0
    return amount if amount > 0 else 0.0


BACKUP_S = 600.0
RESTART_DELAYS_S = (10.0, 30.0, 60.0, 120.0, 300.0)
STOP_FILENAME = "controller.stop"  # in the home: asks `gpurelay up` to stop gracefully


def restore_state(storage: Storage, home: Path) -> bool:
    """Install the backed-up config and job database into an empty ``home``.

    Returns False when there is nothing to restore or ``home`` already has a database (a
    restart that kept its disk: the local copy is newer than any backup)."""
    if (home / STATE_DB_FILENAME).exists():
        return False
    data = storage.get_bytes(STATE_KEY)
    if data is None:
        return False
    home.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gpurelay-restore-") as tmp:
        archive = Path(tmp) / "state.tar.gz"
        archive.write_bytes(data)
        import_state(home, archive, force=True)
    return True


def _content_digest(archive: bytes) -> str:
    """SHA-256 of the archive's member names and contents (not of the gzip/tar framing,
    which carries timestamps)."""
    digest = hashlib.sha256()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            fh = tar.extractfile(member)
            if fh is not None:
                digest.update(member.name.encode() + b"\0" + fh.read() + b"\0")
    return digest.hexdigest()


class StateBackup:
    """Uploads the config and a consistent copy of the job database, but only when they
    changed since the last upload (on the Hub every upload is a commit: 128 per hour)."""

    def __init__(self, storage: Storage, home: Path) -> None:
        self.storage = storage
        self.home = home
        self.digest: str | None = None  # of the copy in storage; None = not known yet

    def __call__(self, *, force: bool = False) -> bool:
        """Back up now if anything changed. False if there was nothing (new) to save."""
        if not (self.home / STATE_DB_FILENAME).exists():
            return False
        with tempfile.TemporaryDirectory(prefix="gpurelay-backup-") as tmp:
            archive = Path(tmp) / "state.tar.gz"
            export_state(self.home, archive)
            data = archive.read_bytes()
        digest = _content_digest(data)
        if self.digest is None and not force:
            stored = self.storage.get_bytes(STATE_KEY)  # a read, not a commit
            self.digest = _content_digest(stored) if stored else ""
        if digest == self.digest and not force:
            return False
        self.storage.put_bytes(data, STATE_KEY)
        self.digest = digest
        return True


def backup_state(storage: Storage, home: Path) -> bool:
    """Upload the config and job database now, changed or not. False if there is no database."""
    return StateBackup(storage, home)(force=True)


def hosted_config(config: Config, env: Mapping[str, str] | None = None) -> Config:
    """The server's own settings: no jobs on its CPU and no desktop to notify.

    Platforms are switched on with the host's environment (there is no setup wizard on a
    server): ``GPURELAY_MODAL=1`` turns Modal on, and setting it is the owner's
    confirmation that they read Modal's terms; its token comes from ``MODAL_TOKEN_ID`` /
    ``MODAL_TOKEN_SECRET``. Without it Modal is off. ``GPURELAY_MODAL_GPU`` picks the GPU."""
    env = os.environ if env is None else env
    update: dict[str, object] = {
        "local": config.backends.local.model_copy(update={"enabled": False})
    }
    if env.get(ENV_MODAL, "").strip().lower() in ("1", "true", "yes"):
        update["modal"] = config.backends.modal.model_copy(update={
            "enabled": True, "terms_reviewed": True,
            "gpu": env.get(ENV_MODAL_GPU, "").strip() or config.backends.modal.gpu,
            "monthly_paid_cap_usd": _usd(env.get(ENV_MODAL_PAID_CAP)),
        })  # fmt: skip
    else:  # the switch decides both ways (a restored backup may say otherwise)
        update["modal"] = config.backends.modal.model_copy(update={"enabled": False})
    backends = config.backends.model_copy(update=update)
    notify = config.notify.model_copy(update={"desktop": False})
    return config.model_copy(update={"backends": backends, "notify": notify})


def prepare_home(home: Path, storage_factory: Callable[[str], Storage]) -> Storage:
    """Restore state if needed, write the server's config, and return the storage.

    ``storage_factory`` opens the Hugging Face repo named by the environment when ``home`` has
    no config yet; otherwise the config names the storage."""
    config_path = home / CONFIG_FILENAME
    repo = os.environ.get(ENV_HF_REPO)
    if not config_path.exists() and not repo:
        raise ConfigError(
            f"no {CONFIG_FILENAME} in {home}: set {ENV_HF_REPO} to the private storage repo "
            "(owner/name) so the server can restore its state from it"
        )
    if config_path.exists():
        config = load_config(config_path)
        if repo and config.storage.hf_repo_id != repo:
            raise ConfigError(f"{config_path} uses another storage repo than {ENV_HF_REPO}={repo}")
        storage = open_storage_from_config(config, home)
    else:
        storage = storage_factory(str(repo))
    if restore_state(storage, home):
        log.info("restored the job database from the storage repo")
    if not config_path.exists():
        raise ConfigError(
            f"the storage repo has no state backup ({STATE_KEY}) yet: run "
            "`gpurelay export-state` on the machine that ran the controller and upload it, "
            "or start the server once with a config.toml in its home"
        )
    save_config(hosted_config(load_config(config_path)), config_path)
    return storage


def watch_stop_file(home: Path, poll_s: float = 0.5) -> threading.Thread:
    """End this process's main thread like Ctrl+C once ``home/controller.stop`` appears.

    The graceful stop that works everywhere: a console-less child on Windows cannot receive
    Ctrl+C or Ctrl+Break. `up` then releases its lease, so the next controller takes over at
    once instead of waiting for the lease to go stale."""
    stop = home / STOP_FILENAME
    # A request written before this process started is a leftover for an earlier controller;
    # one written while it was starting up (a Stop clicked right away) is for this one.
    try:
        import psutil

        started = psutil.Process().create_time()
    except Exception:
        started = time.time()
    try:
        if stop.stat().st_mtime < started - 1.0:
            stop.unlink(missing_ok=True)
    except FileNotFoundError:
        pass

    def watch() -> None:
        while True:
            if stop.exists():
                stop.unlink(missing_ok=True)
                # A real SIGINT (not _thread.interrupt_main): on Windows only that also wakes
                # the main thread from time.sleep at once.
                signal.raise_signal(signal.SIGINT)
                return
            time.sleep(poll_s)

    thread = threading.Thread(target=watch, name="gpurelay-stop-file", daemon=True)
    thread.start()
    return thread


def spawn_controller(cmd: Sequence[str], log_path: Path | None = None) -> subprocess.Popen[bytes]:
    """Start the controller child (on Windows without a console window). Output goes to
    ``log_path`` when given, else to this process's output."""
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        out = open(log_path, "ab")  # noqa: SIM115 - the child owns it from here
        kwargs.update(stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    child: subprocess.Popen[bytes] = subprocess.Popen(list(cmd), **kwargs)  # type: ignore[call-overload]
    return child


class Supervisor:
    """Keeps one controller child process running and backs up the job database.

    :meth:`halt` and :meth:`resume` stop and start the controller on request (the desktop
    app's Stop/Start buttons); :meth:`stop` ends supervision for good."""

    def __init__(
        self,
        command: Sequence[str],
        backup: Callable[[], object],
        *,
        backup_s: float = BACKUP_S,
        spawn: Callable[[Sequence[str]], subprocess.Popen[bytes]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        home: Path | None = None,
    ) -> None:
        self.command = list(command)
        self.home = home  # where the stop file goes (graceful stop on every platform)
        self.backup = backup
        self.backup_s = backup_s
        self.spawn = spawn or spawn_controller
        self.clock = clock
        self.child: subprocess.Popen[bytes] | None = None
        self.enabled = True  # False after halt(): no controller until resume()
        self.restarts = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()  # step() and stop() never interleave
        self._next_start = 0.0
        self._started = 0.0
        self._last_backup = clock()

    def _safe_backup(self) -> None:
        try:
            if self.backup():
                log.info("job database backed up to the storage repo")
        except Exception as exc:  # storage hiccup: keep running, try again next time
            log.warning("job database backup failed: %s", exc)

    def step(self) -> None:
        """One supervision pass: restart a dead controller (with backoff), back up if due."""
        with self._lock:
            if not self._stop.is_set():
                self._step(self.clock())

    def _step(self, now: float) -> None:
        if self.child is not None and self.child.poll() is not None:
            code = self.child.returncode
            self.child = None
            delay = RESTART_DELAYS_S[min(self.restarts, len(RESTART_DELAYS_S) - 1)]
            self.restarts += 1
            self._next_start = now + delay
            log.warning("controller exited with code %s; restarting in %.0f s", code, delay)
            self._safe_backup()
            self._last_backup = now
        if self.child is None and self.enabled and now >= self._next_start:
            self.child = self.spawn(self.command)
            self._started = now
        if self.child is not None and self.restarts and now - self._started > 600:
            self.restarts = 0  # it has been healthy for a while: start the backoff over
        if now - self._last_backup >= self.backup_s:
            self._safe_backup()
            self._last_backup = now

    def run(self, poll_s: float = 5.0) -> None:
        while not self._stop.is_set():
            self.step()
            self._stop.wait(poll_s)

    @property
    def running(self) -> bool:
        child = self.child
        return child is not None and child.poll() is None

    def halt(self, timeout_s: float = 15.0) -> None:
        """Stop the controller and keep it stopped until :meth:`resume`."""
        with self._lock:
            self.enabled = False
            child, self.child = self.child, None
        self._end(child, timeout_s, self.home)

    def resume(self) -> None:
        """Start the controller again at the next step (no backoff)."""
        with self._lock:
            self.enabled = True
            self.restarts = 0
            self._next_start = 0.0

    def stop(self, timeout_s: float = 15.0) -> None:
        """Stop the controller (running GPU sessions continue) and save a last backup. Short
        enough for the host's shutdown grace period (Render: 30 s)."""
        with self._lock:
            self._stop.set()
            child, self.child = self.child, None
        self._end(child, timeout_s, self.home)
        self._safe_backup()

    @staticmethod
    def _end(child: subprocess.Popen[bytes] | None, timeout_s: float, home: Path | None) -> None:
        """Ask ``child`` to stop like Ctrl+C, so `up` releases its lease and the next
        controller takes over at once; kill it if it does not stop in time."""
        if child is None or child.poll() is not None:
            return
        if home is not None:
            (home / STOP_FILENAME).write_text("stop\n", encoding="utf-8")
        elif os.name == "posix":
            child.send_signal(signal.SIGINT)
        else:
            child.terminate()
        try:
            child.wait(timeout_s)
        except subprocess.TimeoutExpired:
            child.kill()


def controller_command(takeover: bool) -> list[str]:
    """``gpurelay up`` with this interpreter (or this executable, in the packaged app)."""
    if getattr(sys, "frozen", False):  # PyInstaller build: the executable runs CLI commands
        cmd = [sys.executable, "up"]
    else:
        cmd = [sys.executable, "-m", "gpurelay", "up"]
    return [*cmd, "--takeover"] if takeover else cmd


def serve(host: str, port: int, home: Path) -> None:  # pragma: no cover - wires real pieces
    import uvicorn

    from gpurelay.controller.service import open_context
    from gpurelay.credentials import hf_token
    from gpurelay.dashboard.app import create_app
    from gpurelay.dashboard.auth import Auth
    from gpurelay.storage.hf_hub import HFHubStorage

    auth = Auth.from_env()  # first: never start without a login
    storage = prepare_home(home, lambda repo: HFHubStorage(repo, hf_token()))
    supervisor = Supervisor(
        controller_command(os.environ.get(ENV_TAKEOVER) == "1"),
        StateBackup(storage, home),
        backup_s=float(os.environ.get(ENV_BACKUP_S) or BACKUP_S),
        home=home,
    )
    thread = threading.Thread(target=supervisor.run, name="gpurelay-supervisor", daemon=True)
    thread.start()
    try:
        ctx = open_context(load_config(home / CONFIG_FILENAME))
        uvicorn.run(create_app(ctx, allowed_hosts=None, auth=auth), host=host, port=port,
                    log_level="warning", proxy_headers=True, forwarded_allow_ips="*")  # fmt: skip
    finally:
        with contextlib.suppress(Exception):
            supervisor.stop()
