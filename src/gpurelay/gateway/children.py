"""Starting, finding and stopping each user's GPU Relay process (plan M3).

A user's process starts when they open their GPU Relay, and when the site starts if they had
jobs in flight (``users.active``). It stops after ``idle_s`` without visits, but never while
one of their jobs still needs its controller: handoffs happen only while it runs. At most
``max_active`` run at once (memory: about 150 MB each); when the site is full, an idle one
is stopped to make room, and otherwise the visitor is asked to come back later.
"""

from __future__ import annotations

import logging
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpurelay.errors import GpuRelayError
from gpurelay.gateway.space import UserSpaces
from gpurelay.gateway.user_server import ENV_LAUNCH_KEY, STOP_NAME
from gpurelay.gateway.userenv import HF, user_environment, write_home

log = logging.getLogger(__name__)

IDLE_S = 30 * 60.0
START_TIMEOUT_S = 90.0
STOP_TIMEOUT_S = 20.0
TERMINAL = frozenset({"completed", "failed_user", "failed_infra", "cancelled"})
NOT_ACTIVE = TERMINAL | {"paused"}  # these need no controller


class SiteFull(GpuRelayError):
    """No room for another user's GPU Relay right now."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Child:
    user_id: str
    port: int
    key: str
    process: Any  # subprocess.Popen
    home: Path
    last_used: float
    active: bool = False  # has jobs that need the controller

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.process.poll() is None


def spawn_user_server(home: Path, env: dict[str, str], port: int) -> Any:
    log_file = open(home / "server.log", "ab")  # noqa: SIM115 - the child owns it from here
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(
        [sys.executable, "-m", "gpurelay.gateway.user_server", "--port", str(port)],
        env=env, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs,
    )  # fmt: skip


def http_get_json(url: str, key: str, timeout: float = 5.0) -> Any:
    import httpx

    from gpurelay.dashboard.desktop_api import LAUNCH_COOKIE

    response = httpx.get(url, headers={"cookie": f"{LAUNCH_COOKIE}={key}"}, timeout=timeout)
    response.raise_for_status()
    return response.json()


@dataclass
class Children:
    spaces: UserSpaces
    max_active: int = 2
    idle_s: float = IDLE_S
    spawn: Callable[[Path, dict[str, str], int], Any] = spawn_user_server
    get_json: Callable[[str, str], Any] = http_get_json
    clock: Callable[[], float] = time.time
    on_jobs: Callable[[str, list[dict[str, Any]]], Any] | None = None  # job emails (M4)
    _children: dict[str, Child] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def home(self, user_id: str) -> Path:
        return self.spaces.root / user_id

    def get(self, user_id: str) -> Child | None:
        with self._lock:
            child = self._children.get(user_id)
            return child if child is not None and child.alive() else None

    def running(self) -> list[Child]:
        with self._lock:
            return [c for c in self._children.values() if c.alive()]

    def jobs_in_flight(self, user_id: str) -> int:
        """How many of the user's jobs are not finished (their GPU Relay must be running)."""
        child = self.get(user_id)
        if child is None:
            return 0
        jobs = self.get_json(child.url + "/api/overview", child.key)["jobs"]
        return sum(1 for j in jobs if j.get("state") not in TERMINAL)

    def can_start(self, user_id: str) -> bool:
        """The user has storage set up (without it their GPU Relay has nowhere to keep jobs)."""
        return HF in self.spaces.saved(user_id) and bool(
            self.spaces.config(user_id).storage.hf_repo_id
        )

    def ensure(self, user_id: str) -> Child:
        """The user's running GPU Relay, started (and waited for) if needed."""
        with self._lock:
            child = self.get(user_id)
            if child is not None:
                child.last_used = self.clock()
                return child
            self._make_room()
            child = self._start(user_id)
        self._wait_ready(child)
        return child

    def _make_room(self) -> None:
        live = self.running()
        if len(live) < self.max_active:
            return
        idle = [c for c in live if not c.active]
        if not idle:
            raise SiteFull("this server is busy with other users' jobs; try again later")
        self._stop(min(idle, key=lambda c: c.last_used))

    def _start(self, user_id: str) -> Child:
        if not self.can_start(user_id):
            raise GpuRelayError("connect Hugging Face storage first (Set up accounts)")
        home = self.home(user_id)
        write_home(home, self.spaces.config(user_id))
        (home / STOP_NAME).unlink(missing_ok=True)
        key, port = secrets.token_urlsafe(32), free_port()
        env = {**user_environment(home, self.spaces.secrets(user_id)),
               **self.spaces.platform_env(user_id), ENV_LAUNCH_KEY: key}  # fmt: skip
        process = self.spawn(home, env, port)
        child = Child(user_id, port, key, process, home, last_used=self.clock())
        self._children[user_id] = child
        log.info("started GPU Relay for %s on port %d", user_id, port)
        return child

    def _wait_ready(self, child: Child) -> None:
        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            if not child.alive():
                raise GpuRelayError("your GPU Relay could not start; see Set up accounts")
            try:
                self.get_json(child.url + "/healthz", child.key)
                return
            except Exception:
                time.sleep(0.5)
        raise GpuRelayError("your GPU Relay is still starting; reload in a minute")

    def _stop(self, child: Child) -> None:
        self._children.pop(child.user_id, None)
        if not child.alive():
            return
        (child.home / STOP_NAME).write_text("stop\n", encoding="utf-8")
        try:
            child.process.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            child.process.kill()
        log.info("stopped GPU Relay for %s", child.user_id)

    def stop(self, user_id: str) -> None:
        with self._lock:
            child = self._children.get(user_id)
            if child is not None:
                self._stop(child)

    def restart(self, user_id: str) -> None:
        """New settings or tokens: a running GPU Relay starts again with them."""
        with self._lock:
            if self.get(user_id) is None:
                return
            self.stop(user_id)
            if self.can_start(user_id):
                self._start(user_id)

    def tick(self) -> None:
        """Note which users have jobs in flight; stop idle ones; restart crashed active ones."""
        now = self.clock()
        with self._lock:
            children = list(self._children.values())
        for child in children:
            if not child.alive():
                with self._lock:
                    self._children.pop(child.user_id, None)
                if child.active and self.can_start(child.user_id):
                    log.warning("GPU Relay for %s exited; starting it again", child.user_id)
                    with self._lock:
                        self._start(child.user_id)
                continue
            try:
                overview = self.get_json(child.url + "/api/overview", child.key)
                jobs = overview["jobs"]
                # An AI fix being worked on or an automatic rerun to make also needs it.
                active = any(j.get("state") not in NOT_ACTIVE for j in jobs) or bool(
                    overview.get("ai_busy")
                )
            except Exception:
                continue  # busy or starting: decide next time
            if self.on_jobs is not None:
                try:
                    self.on_jobs(child.user_id, jobs)
                except Exception as exc:  # a mail problem never stops supervision
                    log.warning("job notices for %s failed: %s", child.user_id, exc)
            if active != child.active:
                child.active = active
                self.spaces.store.set_active(child.user_id, active)
            if not active and now - child.last_used > self.idle_s:
                with self._lock:
                    self._stop(child)

    def start_active(self) -> None:
        """At site start: bring back the GPU Relay of every user with jobs in flight."""
        for user in self.spaces.store.active_users()[: self.max_active]:
            try:
                with self._lock:
                    child = self._start(user.id)
                child.active = True
            except GpuRelayError as exc:
                log.warning("could not start GPU Relay for %s: %s", user.id, exc)

    def stop_all(self) -> None:
        """All at once: the host gives about 30 s to shut down (each saves a last backup)."""
        with self._lock:
            children, self._children = [c for c in self._children.values() if c.alive()], {}
        for child in children:
            (child.home / STOP_NAME).write_text("stop\n", encoding="utf-8")
        deadline = time.monotonic() + STOP_TIMEOUT_S
        for child in children:
            try:
                child.process.wait(max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.process.kill()
