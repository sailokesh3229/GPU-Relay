"""Vast.ai (plan G3): one instance per segment through the REST API
(``https://console.vast.ai/api/v0``). Checked against docs.vast.ai on 2026-10-29
(``docs/verify.md`` #34); not tried on a real account.

* A marketplace: ``POST /bundles/`` finds the cheapest rentable, verified offer of ``gpu``
  (one GPU, reliability at least ``min_reliability``); ``PUT /asks/{offer}/`` rents it with
  the image, the session script as ``onstart`` (at most about 4 KB: the bootstrap travels
  compressed) and the storage token as ``-e HF_TOKEN=...`` (Vast's environment, not the
  command). The offer's ``dph_total`` is the segment's price.
* ``GET /instances/{id}/``: ``actual_status`` null/"loading" (starting), "running",
  "exited"... ``DELETE /instances/{id}/`` destroys it (storage bills until then).
* The session destroys its own instance with the per-instance ``CONTAINER_API_KEY`` Vast
  sets (the docs' ``vastai destroy instance $CONTAINER_ID``); the controller deletes it too.
* Balance: ``GET /users/current/``. VERIFY: one docs page calls the field ``balance``,
  another ``credit``.
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
    STOPPED,
    CloudError,
    Machine,
    Rest,
    Transport,
    urllib_transport,
)
from gpurelay.config import VastSettings

BASE = "https://console.vast.ai/api/v0"
ONSTART_MAX = 4000  # the docs give 4048 characters (and 16 KB for the CLI)
STATES = {"running": RUNNING, "loading": STARTING, "created": STARTING, "exited": STOPPED,
          "stopped": STOPPED, "offline": FAILED, "unknown": FAILED}  # fmt: skip
SELF_END = (
    '"$PY" -m pip install -q vastai >/dev/null 2>&1; '
    'yes | vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" '
    ">/dev/null 2>&1 || true"
)


@dataclass
class VastProvider:
    settings: VastSettings
    secret: Callable[[str], str | None]
    transport: Transport = urllib_transport
    label: str = "Vast.ai"
    vm: bool = False
    self_end: str = SELF_END
    _rest: Rest | None = field(default=None, init=False)

    @property
    def gpu_label(self) -> str:
        return self.settings.gpu

    def missing(self) -> str | None:
        return None if self.secret("api_key") else "add your Vast.ai API key on Your GPU platforms"

    @property
    def rest(self) -> Rest:
        if self._rest is None:
            key = self.secret("api_key")
            if not key:
                raise CloudError("no Vast.ai API key")
            self._rest = Rest(BASE, key, self.label, self.transport)
        return self._rest

    def offers(self) -> list[dict[str, Any]]:
        query: dict[str, Any] = {
            "gpu_name": {"eq": self.settings.gpu},
            "num_gpus": {"eq": 1},
            "rentable": {"eq": True},
            "verified": {"eq": True},
            "reliability": {"gte": self.settings.min_reliability},
            "disk_space": {"gte": self.settings.disk_gb},
            "type": "on-demand",  # VERIFY: the OpenAPI enum says "ondemand"
            "order": [["dph_total", "asc"]],
            "limit": 5,
        }
        if self.settings.max_usd_per_hour:
            query["dph_total"] = {"lte": self.settings.max_usd_per_hour}
        offers: list[dict[str, Any]] = self.rest.call("POST", "/bundles/", query).get("offers", [])
        return offers

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        if len(script) > ONSTART_MAX:
            raise CloudError(f"the session script is {len(script)} characters; Vast.ai takes "
                             f"{ONSTART_MAX}")  # fmt: skip
        for value in env.values():
            if any(c in value for c in " '\"\n"):
                raise CloudError("an environment value cannot be passed to Vast.ai safely")
        offers = self.offers()
        if not offers:
            raise CloudError(f"no Vast.ai offer for one {self.settings.gpu} right now")
        offer = offers[0]
        answer = self.rest.call("PUT", f"/asks/{offer['id']}/", {
            "image": self.settings.image,
            "disk": self.settings.disk_gb,
            "label": name,
            "onstart": script,
            "env": " ".join(f"-e {k}={v}" for k, v in env.items()),
            "runtype": "ssh",  # onstart runs in ssh (or jupyter) mode only
        })  # fmt: skip
        contract = answer.get("new_contract")
        if not answer.get("success") or not contract:
            raise CloudError(f"Vast.ai did not rent offer {offer['id']}: {str(answer)[:200]}")
        out = {"id": str(contract), "offer": str(offer["id"])}
        if offer.get("dph_total"):
            out["usd_per_h"] = repr(float(offer["dph_total"]))
        return out

    def _state(self, instance: dict[str, Any]) -> str:
        status = instance.get("actual_status")
        return STARTING if status is None else STATES.get(str(status), RUNNING)

    def machine(self, machine_id: str) -> Machine:
        try:
            answer = self.rest.call("GET", f"/instances/{machine_id}/")
        except CloudError as exc:
            if exc.status == 404:
                return Machine(machine_id, "", GONE)
            raise
        instance = answer.get("instances")
        if not instance:
            return Machine(machine_id, "", GONE)
        return Machine(machine_id, str(instance.get("label") or ""), self._state(instance),
                       str(instance.get("actual_status")))  # fmt: skip

    def delete(self, machine_id: str) -> None:
        self.rest.call("DELETE", f"/instances/{machine_id}/", ok=(404,))

    def ours(self, prefix: str) -> list[Machine]:
        answer = self.rest.call("GET", "/instances/")
        return [Machine(str(i["id"]), str(i.get("label") or ""), self._state(i))
                for i in answer.get("instances", [])
                if str(i.get("label") or "").startswith(prefix)]  # fmt: skip

    def logs(self, machine_id: str) -> str | None:
        answer = self.rest.call("PUT", f"/instances/request_logs/{machine_id}", {"tail": "2000"})
        url = answer.get("result_url")
        if not url:
            return None
        status, raw = self.transport("GET", str(url), {}, None, 30.0)  # a public S3 link
        return raw.decode("utf-8", "replace") if status == 200 else None

    def price_usd_per_h(self) -> float | None:
        offers = self.offers()
        return float(offers[0]["dph_total"]) if offers and offers[0].get("dph_total") else None

    def balance_usd(self) -> float | None:
        user = self.rest.call("GET", "/users/current/")
        for key in ("credit", "balance"):
            if isinstance(user.get(key), (int, float)):
                return float(user[key])
        return None
