"""DigitalOcean (plan G4): one GPU Droplet per segment through the API
(``https://api.digitalocean.com/v2``). Checked against docs.digitalocean.com on 2026-10-29
(``docs/verify.md`` #36); not tried on a real account.

* ``POST /droplets`` with ``name``, ``region``, ``size`` (a GPU plan slug), ``image``
  (``gpu-h100x1-base``: the docs say it is for every single-GPU plan), ``tags`` and the
  session script as ``user_data`` (at most 64 KiB; it holds the storage token, which only the
  account can read). ``GET /droplets/{id}``: ``status`` new | active | off | archive.
  ``DELETE /droplets/{id}``.
* A powered-off GPU Droplet still bills: the session powers off when done (so the controller
  notices), and the controller destroys the Droplet.
* GPU Droplets are listed with ``?type=gpus`` (a tag filter may leave them out).
* Price: ``price_hourly`` of ``GET /sizes``. Balance: the docs do not say how credit shows
  in ``/customers/my/balance``, so credit left is estimated from what you entered.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from gpurelay.backends.cloud import (
    FAILED,
    GONE,
    RUNNING,
    STARTING,
    STOPPED,
    CloudError,
    Machine,
    Rest,
    Transport,
    urllib_transport,
)
from gpurelay.config import DigitalOceanSettings

BASE = "https://api.digitalocean.com/v2"
USER_DATA_MAX = 64 * 1024
STATES = {"new": STARTING, "active": RUNNING, "off": STOPPED, "archive": GONE}


def droplet_name(name: str) -> str:
    return re.sub(r"[^a-z0-9.-]", "-", name.lower())[:255]


@dataclass
class DigitalOceanProvider:
    settings: DigitalOceanSettings
    secret: Callable[[str], str | None]
    transport: Transport = urllib_transport
    label: str = "DigitalOcean"
    vm: bool = True
    self_end: str = "shutdown -h now"
    _rest: Rest | None = field(default=None, init=False)

    @property
    def gpu_label(self) -> str:
        return self.settings.size

    def missing(self) -> str | None:
        if self.secret("token"):
            return None
        return "add your DigitalOcean token on Your GPU platforms"

    @property
    def rest(self) -> Rest:
        if self._rest is None:
            token = self.secret("token")
            if not token:
                raise CloudError("no DigitalOcean token")
            self._rest = Rest(BASE, token, self.label, self.transport)
        return self._rest

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        if len(script.encode("utf-8")) > USER_DATA_MAX:
            raise CloudError("the session script is over DigitalOcean's 64 KiB user data limit")
        answer = self.rest.call("POST", "/droplets", {
            "name": droplet_name(name),
            "region": self.settings.region,
            "size": self.settings.size,
            "image": self.settings.image,
            "user_data": script,
            "tags": ["gpurelay"],
        })  # fmt: skip
        droplet = answer.get("droplet") or {}
        if not droplet.get("id"):
            raise CloudError(f"DigitalOcean did not return a Droplet id: {str(answer)[:200]}")
        return {"id": str(droplet["id"])}

    def machine(self, machine_id: str) -> Machine:
        try:
            droplet = self.rest.call("GET", f"/droplets/{machine_id}").get("droplet") or {}
        except CloudError as exc:
            if exc.status == 404:
                return Machine(machine_id, "", GONE)
            raise
        status = str(droplet.get("status", ""))
        return Machine(machine_id, str(droplet.get("name") or ""), STATES.get(status, FAILED),
                       status)  # fmt: skip

    def delete(self, machine_id: str) -> None:
        self.rest.call("DELETE", f"/droplets/{machine_id}", ok=(404,))

    def ours(self, prefix: str) -> list[Machine]:
        prefix = droplet_name(prefix)
        answer = self.rest.call("GET", "/droplets?type=gpus&per_page=200")
        return [Machine(str(d["id"]), str(d.get("name") or ""),
                        STATES.get(str(d.get("status")), FAILED))
                for d in answer.get("droplets", [])
                if str(d.get("name") or "").startswith(prefix)]  # fmt: skip

    def logs(self, machine_id: str) -> str | None:
        return None  # no console API; the job's own log is in storage

    def price_usd_per_h(self) -> float | None:
        for size in self.rest.call("GET", "/sizes?per_page=200").get("sizes", []):
            if size.get("slug") == self.settings.size and size.get("price_hourly"):
                return float(size["price_hourly"])
        return None

    def balance_usd(self) -> float | None:
        return None  # how credit shows in /customers/my/balance is not documented
