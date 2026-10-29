"""Lambda (plan G4): one instance per segment through the Cloud API
(``https://cloud.lambda.ai/api/v1``, OpenAPI 1.10.0). Checked against docs.lambda.ai on
2026-10-29 (``docs/verify.md`` #35); not tried on a real account.

* ``POST /instance-operations/launch`` with ``region_name``, ``instance_type_name``,
  ``ssh_key_names`` (exactly one is required: GPU Relay adds a key named "gpurelay" once and
  never keeps its private half), ``name`` and the session script as ``user_data``
  (cloud-init). The storage token is in that script: Lambda has no other way to give a
  machine its environment, and only the account can read it.
* ``GET /instances/{id}``: ``status`` booting | active | unhealthy | terminating |
  terminated | preempted. ``POST /instance-operations/terminate`` ends it.
* Lambda bills until an instance is terminated, and a shutdown from inside does not stop it
  (it only puts the instance in "Alert" status), so the session does not shut down: the
  controller terminates the instance as soon as the segment is over.
* Price: ``price_cents_per_hour`` of ``GET /instance-types``. Lambda reports no balance, so
  credit left is estimated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gpurelay.backends.cloud import (
    FAILED,
    GONE,
    RUNNING,
    STARTING,
    CloudError,
    Machine,
    Rest,
    Transport,
    urllib_transport,
)
from gpurelay.config import LambdaSettings

BASE = "https://cloud.lambda.ai/api/v1"
SSH_KEY = "gpurelay"
STATES = {"booting": STARTING, "active": RUNNING, "unhealthy": RUNNING,
          "terminating": GONE, "terminated": GONE, "preempted": GONE}  # fmt: skip


@dataclass
class LambdaProvider:
    settings: LambdaSettings
    secret: Callable[[str], str | None]
    transport: Transport = urllib_transport
    label: str = "Lambda"
    vm: bool = True
    self_end: str = ""  # a shutdown keeps billing: the controller terminates the instance
    _rest: Rest | None = field(default=None, init=False)

    @property
    def gpu_label(self) -> str:
        return self.settings.instance_type

    def missing(self) -> str | None:
        return None if self.secret("api_key") else "add your Lambda API key on Your GPU platforms"

    @property
    def rest(self) -> Rest:
        if self._rest is None:
            key = self.secret("api_key")
            if not key:
                raise CloudError("no Lambda API key")
            self._rest = Rest(BASE, key, self.label, self.transport)
        return self._rest

    def _types(self) -> dict[str, Any]:
        data: dict[str, Any] = self.rest.call("GET", "/instance-types").get("data", {})
        return data

    def _ssh_key(self) -> None:
        names = {k.get("name") for k in self.rest.call("GET", "/ssh-keys").get("data", [])}
        if SSH_KEY not in names:  # Lambda generates the pair; its private half is dropped
            self.rest.call("POST", "/ssh-keys", {"name": SSH_KEY})

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        kind = self._types().get(self.settings.instance_type)
        if kind is None:
            raise CloudError(f"Lambda has no instance type {self.settings.instance_type!r}")
        regions = [r["name"] for r in kind.get("regions_with_capacity_available", [])]
        region = self.settings.region or (regions[0] if regions else None)
        if region is None or (self.settings.region and region not in regions):
            raise CloudError(f"no Lambda capacity for {self.settings.instance_type} right now")
        self._ssh_key()
        answer = self.rest.call("POST", "/instance-operations/launch", {
            "region_name": region,
            "instance_type_name": self.settings.instance_type,
            "ssh_key_names": [SSH_KEY],
            "name": name[:64],
            "user_data": script,
        })  # fmt: skip
        ids = (answer.get("data") or {}).get("instance_ids") or []
        if not ids:
            raise CloudError(f"Lambda did not return an instance id: {str(answer)[:200]}")
        out = {"id": str(ids[0]), "region": region}
        cents = (kind.get("instance_type") or {}).get("price_cents_per_hour")
        if cents:
            out["usd_per_h"] = repr(float(cents) / 100)
        return out

    def machine(self, machine_id: str) -> Machine:
        try:
            data = self.rest.call("GET", f"/instances/{machine_id}").get("data") or {}
        except CloudError as exc:
            if exc.status == 404:
                return Machine(machine_id, "", GONE)
            raise
        status = str(data.get("status", ""))
        return Machine(machine_id, str(data.get("name") or ""), STATES.get(status, FAILED), status)

    def delete(self, machine_id: str) -> None:
        self.rest.call("POST", "/instance-operations/terminate", {"instance_ids": [machine_id]},
                       ok=(404,))  # fmt: skip

    def ours(self, prefix: str) -> list[Machine]:
        return [Machine(str(i["id"]), str(i.get("name") or ""),
                        STATES.get(str(i.get("status")), FAILED))
                for i in self.rest.call("GET", "/instances").get("data", [])
                if str(i.get("name") or "").startswith(prefix)]  # fmt: skip

    def logs(self, machine_id: str) -> str | None:
        return None  # no console API; the job's own log is in storage

    def price_usd_per_h(self) -> float | None:
        kind = self._types().get(self.settings.instance_type) or {}
        cents = (kind.get("instance_type") or {}).get("price_cents_per_hour")
        return float(cents) / 100 if cents else None

    def balance_usd(self) -> float | None:
        return None  # not in the API: estimated from the credit you entered
