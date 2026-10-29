"""One controller per *store*, across machines (Phase 9 item 6: always-on controller on a VM).

The OS lock in :mod:`gpurelay.controller.lock` stops two controllers on one machine. Moving
the controller to a VM adds the risk of one on the laptop and one on the VM, both launching
the same jobs. The **lease** lives in the shared storage (``controller/lease.json``)::

    {"holder": "<host>:<pid>:<random>", "seq": 41, "renewed_at": "<holder's clock>"}

* The holder renews it every ``renew_s`` (the sequence number changes).
* A new controller takes a free lease, takes over one whose ``renewed_at`` is older than
  ``stale_s``, and otherwise watches it for up to ``1.5 x renew_s``: if it changes, another
  controller is alive and the new one refuses to start (``--takeover`` skips the wait).
* A holder that finds someone else's lease when renewing stops (it was taken over).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from gpurelay.errors import ControllerRunningError, GpuRelayError
from gpurelay.storage.base import Storage

log = logging.getLogger(__name__)

LEASE_KEY = "controller/lease.json"
RENEW_S = 600.0  # a renewal is a storage write (a commit on the Hub: 128 per hour per repo)
STALE_S = 1800.0  # by the holder's own clock: generous, so clock skew cannot hurt


class LeaseLostError(GpuRelayError):
    """Another controller took over the lease: this one must stop."""


def new_holder_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


@dataclass
class ControllerLease:
    storage: Storage
    holder: str = field(default_factory=new_holder_id)
    renew_s: float = RENEW_S
    stale_s: float = STALE_S
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    _seq: int = 0
    _renewed: float = 0.0

    def _read(self) -> dict[str, object] | None:
        raw = self.storage.get_bytes(LEASE_KEY)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _write(self) -> None:
        self._seq += 1
        body = {
            "holder": self.holder,
            "seq": self._seq,
            "renewed_at": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
        }
        self.storage.put_bytes(json.dumps(body).encode(), LEASE_KEY)
        self._renewed = self.clock()

    def acquire(self, *, takeover: bool = False) -> None:
        """Take the lease, or raise :class:`ControllerRunningError` if a live one exists."""
        current = self._read()
        other = current is not None and current.get("holder") != self.holder
        if other and not takeover and current is not None and not self._is_stale(current):
            log.info("another controller (%s) holds the lease; checking it is alive...",
                     current.get("holder"))  # fmt: skip
            deadline = self.clock() + 1.5 * self.renew_s
            while self.clock() < deadline:
                self.sleep(min(10.0, self.renew_s / 4))
                latest = self._read()
                if latest is None:
                    break  # released: free to take
                if latest.get("seq") != current.get("seq") or (
                    latest.get("holder") != current.get("holder")
                ):
                    raise ControllerRunningError(
                        f"another controller is running ({latest.get('holder')}) on "
                        "this store; stop it first, or start with --takeover"
                    )
        if current is not None and current.get("holder") != self.holder:
            log.warning("taking over the controller lease from %s", current.get("holder"))
        self._write()

    def _is_stale(self, lease: dict[str, object]) -> bool:
        try:
            # Python 3.10 refuses a trailing "Z" (UTC): the lease would always look stale
            stamp = str(lease["renewed_at"]).replace("Z", "+00:00")
            renewed = datetime.fromisoformat(stamp).timestamp()
        except (KeyError, ValueError):
            return True
        return self.clock() - renewed > self.stale_s

    def renew(self, *, force: bool = False) -> None:
        """Check the lease every call (a read) and rewrite it when due (a commit). Raises
        :class:`LeaseLostError` as soon as another controller took over, so two controllers
        never act together for longer than one tick (e.g. during a server redeploy)."""
        current = self._read()
        if current is not None and current.get("holder") != self.holder:
            raise LeaseLostError(f"controller lease taken over by {current.get('holder')}")
        if not force and self.clock() - self._renewed < self.renew_s:
            return
        self._write()

    def release(self) -> None:
        current = self._read()
        if current is not None and current.get("holder") == self.holder:
            self.storage.delete(LEASE_KEY)
