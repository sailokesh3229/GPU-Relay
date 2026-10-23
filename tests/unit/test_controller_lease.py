"""Cross-machine single-controller lease (Phase 9 item 6), two "machines" on one store."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from gpurelay.controller.lease import LEASE_KEY, ControllerLease, LeaseLostError
from gpurelay.errors import ControllerRunningError
from gpurelay.storage import LocalFSStorage


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _lease(
    storage: LocalFSStorage, holder: str, clock: Clock, sleep: Callable[[float], None] | None = None
) -> ControllerLease:
    def advance(seconds: float) -> None:
        clock.now += seconds

    return ControllerLease(storage, holder=holder, renew_s=120, stale_s=900, clock=clock,
                           sleep=sleep or advance)  # fmt: skip


def test_free_lease_is_taken_and_released(tmp_path: Path) -> None:
    storage, clock = LocalFSStorage(tmp_path), Clock()
    laptop = _lease(storage, "laptop", clock)
    laptop.acquire()
    assert json.loads(storage.get_bytes(LEASE_KEY) or b"")["holder"] == "laptop"
    laptop.release()
    assert storage.get_bytes(LEASE_KEY) is None


def test_live_holder_blocks_a_second_controller(tmp_path: Path) -> None:
    storage, clock = LocalFSStorage(tmp_path), Clock()
    vm = _lease(storage, "vm", clock)
    vm.acquire()

    def vm_keeps_renewing(_seconds: float) -> None:
        clock.now += 30
        vm.renew(force=True)

    laptop = _lease(storage, "laptop", clock, sleep=vm_keeps_renewing)
    with pytest.raises(ControllerRunningError, match="vm"):
        laptop.acquire()
    assert json.loads(storage.get_bytes(LEASE_KEY) or b"")["holder"] == "vm"


def test_silent_or_stale_holder_is_taken_over_and_then_stops(tmp_path: Path) -> None:
    storage, clock = LocalFSStorage(tmp_path), Clock()
    vm = _lease(storage, "vm", clock)
    vm.acquire()
    # The VM went quiet (no renewals while the laptop watches): the laptop takes over.
    laptop = _lease(storage, "laptop", clock)
    laptop.acquire()
    assert json.loads(storage.get_bytes(LEASE_KEY) or b"")["holder"] == "laptop"
    clock.now += 200
    with pytest.raises(LeaseLostError, match="laptop"):
        vm.renew()  # the VM comes back: it must stop
    vm.release()  # not its lease anymore: left alone
    assert storage.get_bytes(LEASE_KEY) is not None
    # A lease not renewed for longer than stale_s is taken without waiting.
    clock.now += 2000
    waits: list[float] = []
    _lease(storage, "new", clock, sleep=waits.append).acquire()
    assert waits == []  # stale: no watching


def test_takeover_skips_the_wait(tmp_path: Path) -> None:
    storage, clock = LocalFSStorage(tmp_path), Clock()
    _lease(storage, "vm", clock).acquire()
    waits: list[float] = []
    _lease(storage, "laptop", clock, sleep=waits.append).acquire(takeover=True)
    assert waits == []
    assert json.loads(storage.get_bytes(LEASE_KEY) or b"")["holder"] == "laptop"


def test_renew_is_rate_limited(tmp_path: Path) -> None:
    storage, clock = LocalFSStorage(tmp_path), Clock()
    lease = _lease(storage, "a", clock)
    lease.acquire()
    seq = json.loads(storage.get_bytes(LEASE_KEY) or b"")["seq"]
    clock.now += 60
    lease.renew()
    assert json.loads(storage.get_bytes(LEASE_KEY) or b"")["seq"] == seq  # not due yet
    clock.now += 61
    lease.renew()
    assert json.loads(storage.get_bytes(LEASE_KEY) or b"")["seq"] == seq + 1
