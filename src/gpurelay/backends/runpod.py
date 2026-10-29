"""RunPod (plan G3): one Pod per segment through REST API v2 (``https://api.runpod.io/v2``;
v1 retires on 15 Nov 2026). Checked against docs.runpod.io on 2026-10-29 (``docs/verify.md``
#33); not tried on a real account.

* ``POST /v2/pods`` with the image, one GPU of ``gpu`` (a catalog id), ``env`` (the storage
  token: RunPod's own way to give a container its environment, never in the command) and the
  session script as the container command. ``GET /v2/pods/{id}``: ``status`` PROVISIONING |
  STARTING | RUNNING | EXITED | ERROR | TERMINATED. ``DELETE /v2/pods/{id}`` terminates.
* Price: ``GET /v2/catalog/gpus`` (per GPU per hour, secure or community cloud).
  Balance: GraphQL ``myself { clientBalance }``.
* The session stops its own Pod when done (``RUNPOD_POD_ID`` and the Pod-scoped
  ``RUNPOD_API_KEY`` that RunPod sets); VERIFY: what that key may do is not documented, so
  the controller deletes the Pod in any case. A stopped Pod bills only for its disk.
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
from gpurelay.config import RunPodSettings

BASE = "https://api.runpod.io"
GRAPHQL = "https://api.runpod.io/graphql"
STATES = {"PROVISIONING": STARTING, "STARTING": STARTING, "RUNNING": RUNNING,
          "EXITED": STOPPED, "TERMINATED": GONE, "ERROR": FAILED}  # fmt: skip
SELF_END = """\
"$PY" - <<'GPURELAY_END' >/dev/null 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" >/dev/null 2>&1
import json, os, urllib.request as u
pod = os.environ["RUNPOD_POD_ID"]
u.urlopen(u.Request(f"https://api.runpod.io/v2/pods/{pod}/action", method="POST",
                    data=json.dumps({"action": "stop"}).encode(),
                    headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
                             "Content-Type": "application/json"}), timeout=30)
GPURELAY_END"""


@dataclass
class RunPodProvider:
    settings: RunPodSettings
    secret: Callable[[str], str | None]  # field key -> value ("api_key")
    transport: Transport = urllib_transport
    label: str = "RunPod"
    vm: bool = False
    self_end: str = SELF_END
    _rest: Rest | None = field(default=None, init=False)

    @property
    def gpu_label(self) -> str:
        return self.settings.gpu

    def missing(self) -> str | None:
        return None if self.secret("api_key") else "add your RunPod API key on Your GPU platforms"

    @property
    def rest(self) -> Rest:
        if self._rest is None:
            key = self.secret("api_key")
            if not key:
                raise CloudError("no RunPod API key")
            self._rest = Rest(BASE, key, self.label, self.transport)
        return self._rest

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        pod = self.rest.call("POST", "/v2/pods", {
            "name": name,
            "image": self.settings.image,
            "gpu": {"id": self.settings.gpu, "count": 1},
            "cloud": self.settings.cloud,
            "disk": self.settings.disk_gb,
            "env": dict(env),
            "entrypoint": ["/bin/bash", "-c"],
            "cmd": [script],
        })  # fmt: skip
        if not pod.get("id"):
            raise CloudError(f"RunPod did not return a Pod id: {str(pod)[:200]}")
        out = {"id": str(pod["id"])}
        if pod.get("cost"):
            out["usd_per_h"] = repr(float(pod["cost"]))
        return out

    def machine(self, machine_id: str) -> Machine:
        try:
            pod = self.rest.call("GET", f"/v2/pods/{machine_id}")
        except CloudError as exc:
            if exc.status == 404:
                return Machine(machine_id, "", GONE)
            raise
        status = str(pod.get("status"))
        return Machine(machine_id, str(pod.get("name", "")), STATES.get(status, RUNNING), status)

    def delete(self, machine_id: str) -> None:
        self.rest.call("DELETE", f"/v2/pods/{machine_id}", ok=(404,))

    def ours(self, prefix: str) -> list[Machine]:
        found: list[Machine] = []
        cursor = None
        for _ in range(20):
            path = "/v2/pods?limit=1000" + (f"&cursor={cursor}" if cursor else "")
            page: dict[str, Any] = self.rest.call("GET", path)
            for pod in page.get("pods", []):
                if str(pod.get("name", "")).startswith(prefix):
                    found.append(Machine(str(pod["id"]), str(pod["name"]),
                                         STATES.get(str(pod.get("status")), RUNNING)))  # fmt: skip
            more = page.get("pagination") or {}
            cursor = more.get("nextCursor")
            if not more.get("hasNextPage") or not cursor:
                break
        return found

    def logs(self, machine_id: str) -> str | None:
        return None  # an event stream; the job's own log is in storage

    def price_usd_per_h(self) -> float | None:
        catalog = self.rest.call("GET", "/v2/catalog/gpus")
        for gpu in catalog.get("gpus", []):
            if gpu.get("id") == self.settings.gpu:
                price = (gpu.get("price") or {}).get(self.settings.cloud.lower())
                return float(price) if price else None
        return None

    def balance_usd(self) -> float | None:
        answer = self.rest.call("POST", GRAPHQL, {"query": "query { myself { clientBalance } }"})
        value = ((answer.get("data") or {}).get("myself") or {}).get("clientBalance")
        return float(value) if value is not None else None
