"""Google Cloud (plan G5): one Compute Engine GPU VM per segment through the REST API
(``https://compute.googleapis.com/compute/v1``), authorised with the user's service account
key (google-auth, the ``gcp`` extra). Checked against cloud.google.com on 2026-10-29
(``docs/verify.md`` #38); the request body of ``instances.insert`` could not be read there,
so it follows the GPU guide's REST example. Not tried on a real account.

* ``instances.insert``: ``machine_type`` with one ``accelerator`` (GPUs need
  ``onHostMaintenance: TERMINATE``), a Deep Learning VM image (``image_project`` /
  ``image_family``) with ``install-nvidia-driver=True``, the session script as the
  ``startup-script`` metadata item (it holds the storage token, readable by the project
  only), an external IP for internet access, and the label ``gpurelay=<owner>``.
* ``instances.get`` ``status``: PROVISIONING | STAGING | RUNNING | STOPPING | SUSPENDING |
  SUSPENDED | REPAIRING | TERMINATED. ``instances.delete``. The session shuts the VM down
  when done (TERMINATED) and the controller deletes it (disks bill until then).
* Free trial accounts cannot add GPUs; new projects need GPU quota (NVIDIA_T4_GPUS and
  GPUs (all regions)).
* There is no simple price API and Google shows trial credit only in its console: the price
  is ``usd_per_hour`` or the estimate below, and credit left is estimated.
"""

from __future__ import annotations

import json
import re
import time
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
from gpurelay.config import GcpSettings

BASE = "https://compute.googleapis.com/compute/v1"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
STATES = {"PROVISIONING": STARTING, "STAGING": STARTING, "RUNNING": RUNNING,
          "REPAIRING": RUNNING, "STOPPING": STOPPED, "SUSPENDING": STOPPED,
          "SUSPENDED": STOPPED, "TERMINATED": STOPPED}  # fmt: skip
# VERIFY on cloud.google.com/compute/gpus-pricing: on-demand us-central1, USD per hour.
ESTIMATES = {("n1-standard-4", "nvidia-tesla-t4"): 0.19 + 0.35}


def vm_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]", "-", name.lower())
    return cleaned[:63].rstrip("-")


def google_token(info: dict[str, Any]) -> tuple[str, float]:
    """An access token for the service account key ``info``, and when it expires."""
    from google.auth.transport.requests import Request  # optional extra: gpurelay[gcp]
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_info(info, scopes=[SCOPE])
    credentials.refresh(Request())
    expiry = credentials.expiry.timestamp() if credentials.expiry else time.time() + 1800
    return str(credentials.token), expiry


@dataclass
class GcpProvider:
    settings: GcpSettings
    secret: Callable[[str], str | None]
    transport: Transport = urllib_transport
    token_source: Callable[[dict[str, Any]], tuple[str, float]] = google_token
    label: str = "Google Cloud"
    vm: bool = True
    self_end: str = "shutdown -h now"
    _token: tuple[str, float] | None = field(default=None, init=False)

    @property
    def gpu_label(self) -> str:
        return self.settings.accelerator

    def _key(self) -> dict[str, Any]:
        raw = self.secret("service_account_key")
        if not raw:
            raise CloudError("no Google Cloud service account key")
        info: dict[str, Any] = json.loads(raw)
        return info

    @property
    def project(self) -> str:
        project = self.secret("project_id") or self._key().get("project_id")
        if not project:
            raise CloudError("no Google Cloud project ID")
        return str(project)

    def missing(self) -> str | None:
        if not self.secret("service_account_key"):
            return "add your Google Cloud service account key on Your GPU platforms"
        import importlib.util

        if self.token_source is google_token and importlib.util.find_spec("google.auth") is None:
            return "pip install 'gpurelay[gcp]'"
        return None

    @property
    def rest(self) -> Rest:
        if self._token is None or self._token[1] - time.time() < 300:
            self._token = self.token_source(self._key())
        zone = self.settings.zone
        return Rest(f"{BASE}/projects/{self.project}/zones/{zone}", self._token[0], self.label,
                    self.transport)  # fmt: skip

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        zone, s = self.settings.zone, self.settings
        name = vm_name(name)
        owner = name.split("-")[1]
        self.rest.call("POST", "/instances", {
            "name": name,
            "machineType": f"zones/{zone}/machineTypes/{s.machine_type}",
            "guestAccelerators": [{"acceleratorType": f"zones/{zone}/acceleratorTypes/"
                                                      f"{s.accelerator}", "acceleratorCount": 1}],
            "scheduling": {"onHostMaintenance": "TERMINATE", "automaticRestart": False},
            "disks": [{"boot": True, "autoDelete": True, "type": "PERSISTENT",
                       "initializeParams": {"diskSizeGb": str(s.disk_gb), "sourceImage":
                           f"projects/{s.image_project}/global/images/family/{s.image_family}"}}],
            "networkInterfaces": [{"network": "global/networks/default",
                                   "accessConfigs": [{"type": "ONE_TO_ONE_NAT",
                                                      "name": "External NAT"}]}],
            "metadata": {"items": [{"key": "startup-script", "value": script},
                                   {"key": "install-nvidia-driver", "value": "True"}]},
            "labels": {"gpurelay": owner},
        })  # fmt: skip
        return {"id": name}  # instances are addressed by name

    def machine(self, machine_id: str) -> Machine:
        try:
            vm = self.rest.call("GET", f"/instances/{machine_id}")
        except CloudError as exc:
            if exc.status == 404:
                return Machine(machine_id, machine_id, GONE)
            raise
        status = str(vm.get("status", ""))
        return Machine(machine_id, machine_id, STATES.get(status, FAILED), status)

    def delete(self, machine_id: str) -> None:
        self.rest.call("DELETE", f"/instances/{machine_id}", ok=(404,))

    def ours(self, prefix: str) -> list[Machine]:
        owner = prefix.split("-")[1]
        answer = self.rest.call("GET", f"/instances?filter=labels.gpurelay%3D{owner}")
        return [Machine(str(vm["name"]), str(vm["name"]),
                        STATES.get(str(vm.get("status")), FAILED))
                for vm in answer.get("items", [])
                if str(vm.get("name", "")).startswith(vm_name(prefix))]  # fmt: skip

    def logs(self, machine_id: str) -> str | None:
        answer = self.rest.call("GET", f"/instances/{machine_id}/serialPort?port=1")
        return str(answer.get("contents") or "") or None

    def price_usd_per_h(self) -> float | None:
        return ESTIMATES.get((self.settings.machine_type, self.settings.accelerator))

    def balance_usd(self) -> float | None:
        return None  # trial credit is shown only in Google's console: estimated
