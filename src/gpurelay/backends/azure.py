"""Azure (plan G5): one GPU virtual machine per segment through the Azure SDK for Python
(azure-identity, azure-mgmt-compute, azure-mgmt-network, azure-mgmt-resource: the ``azure``
extra), with the user's service principal. Checked against learn.microsoft.com on
2026-10-29 (``docs/verify.md`` #39): power states and billing, Spot, the NVIDIA driver
extension, quotas. The SDK's quickstart page could not be read, so the calls below follow
the SDK's published models (VERIFY on a real subscription). Not tried on a real account.

* Everything lives in one resource group (``resource_group``) holding only GPU Relay's
  machines: a virtual network (made once), then per segment a public IP, a network
  interface and the VM, all set to be deleted with the VM. The session script is the VM's
  ``custom_data`` (cloud-init; it holds the storage token, readable by the subscription
  only). The image (``image``) must have NVIDIA drivers.
* A VM shut down from inside stays allocated and keeps billing, so the session does not
  shut down: the controller deletes the VM as soon as the segment is over.
* Free trial subscriptions cannot get quota increases; GPU families start with little or no
  vCPU quota (Quotas > Compute in the portal).
* Price: the public Azure retail prices API (Linux, pay as you go). No credit API: credit
  left is estimated.
"""

from __future__ import annotations

import base64
import json
import urllib.parse
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
    Transport,
    urllib_transport,
)
from gpurelay.config import AzureSettings

PRICES = "https://prices.azure.com/api/retail/prices"
VNET, SUBNET = "gpurelay-vnet", "default"
POWER = {"PowerState/starting": STARTING, "PowerState/running": RUNNING,
         "PowerState/stopping": STOPPED, "PowerState/stopped": STOPPED,
         "PowerState/deallocating": STOPPED, "PowerState/deallocated": STOPPED}  # fmt: skip


def azure_clients(tenant: str, client: str, secret: str, subscription: str) -> dict[str, Any]:
    from azure.identity import ClientSecretCredential  # optional extra: gpurelay[azure]
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.network import NetworkManagementClient
    from azure.mgmt.resource import ResourceManagementClient

    credential = ClientSecretCredential(tenant, client, secret)
    return {
        "compute": ComputeManagementClient(credential, subscription),
        "network": NetworkManagementClient(credential, subscription),
        "resource": ResourceManagementClient(credential, subscription),
    }


def ssh_public_key() -> str:
    """A fresh key for the VM's admin account (Linux VMs need one); the private half is never
    kept, so nobody can log in."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    return (
        key.public_key()
        .public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
        .decode("ascii")
    )


def vm_name(name: str) -> str:
    return "".join(c if c.isalnum() or c == "-" else "-" for c in name.lower())[:64]


@dataclass
class AzureProvider:
    settings: AzureSettings
    secret: Callable[[str], str | None]
    clients_factory: Callable[[str, str, str, str], dict[str, Any]] = azure_clients
    key_source: Callable[[], str] = ssh_public_key
    transport: Transport = urllib_transport
    label: str = "Azure"
    vm: bool = True
    self_end: str = ""  # a shutdown keeps billing: the controller deletes the VM
    _clients: dict[str, Any] | None = field(default=None, init=False)
    _network_ready: bool = field(default=False, init=False)

    @property
    def gpu_label(self) -> str:
        return self.settings.vm_size

    def missing(self) -> str | None:
        if not all(self.secret(k) for k in ("tenant_id", "client_id", "client_secret",
                                            "subscription_id")):  # fmt: skip
            return "add your Azure service principal on Your GPU platforms"
        import importlib.util

        if self.clients_factory is azure_clients and importlib.util.find_spec("azure") is None:
            return "pip install 'gpurelay[azure]'"
        return None

    @property
    def clients(self) -> dict[str, Any]:
        if self._clients is None:
            values = [self.secret(k) for k in ("tenant_id", "client_id", "client_secret",
                                               "subscription_id")]  # fmt: skip
            if not all(values):
                raise CloudError("no Azure service principal")
            self._clients = self.clients_factory(*[str(v) for v in values])
        return self._clients

    def _network(self) -> str:
        """The resource group and virtual network (made once); returns the subnet id."""
        rg, location = self.settings.resource_group, self.settings.location
        network = self.clients["network"]
        if not self._network_ready:
            self.clients["resource"].resource_groups.create_or_update(rg, {"location": location})
            network.virtual_networks.begin_create_or_update(rg, VNET, {
                "location": location,
                "address_space": {"address_prefixes": ["10.42.0.0/16"]},
                "subnets": [{"name": SUBNET, "address_prefix": "10.42.0.0/24"}],
            }).result()  # fmt: skip
            self._network_ready = True
        return str(network.subnets.get(rg, VNET, SUBNET).id)

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        s, rg = self.settings, self.settings.resource_group
        name = vm_name(name)
        publisher, offer, sku, version = s.image.split(":")
        subnet = self._network()
        network = self.clients["network"]
        ip = network.public_ip_addresses.begin_create_or_update(rg, f"{name}-ip", {
            "location": s.location, "sku": {"name": "Standard"},
            "public_ip_allocation_method": "Static",
        }).result()  # fmt: skip
        nic = network.network_interfaces.begin_create_or_update(rg, f"{name}-nic", {
            "location": s.location,
            "ip_configurations": [{"name": "primary", "subnet": {"id": subnet},
                                   "public_ip_address": {"id": ip.id, "delete_option": "Delete"}}],
        }).result()  # fmt: skip
        self.clients["compute"].virtual_machines.begin_create_or_update(rg, name, {
            "location": s.location,
            "tags": {"gpurelay": name.split("-")[1]},
            "hardware_profile": {"vm_size": s.vm_size},
            "storage_profile": {
                "image_reference": {"publisher": publisher, "offer": offer, "sku": sku,
                                    "version": version},
                "os_disk": {"create_option": "FromImage", "delete_option": "Delete",
                            "disk_size_gb": s.disk_gb},
            },
            "os_profile": {
                "computer_name": name[:64], "admin_username": "gpurelay",
                "custom_data": base64.b64encode(script.encode("utf-8")).decode("ascii"),
                "linux_configuration": {"disable_password_authentication": True, "ssh": {
                    "public_keys": [{"path": "/home/gpurelay/.ssh/authorized_keys",
                                     "key_data": self.key_source()}]}},
            },
            "network_profile": {"network_interfaces": [{"id": nic.id, "primary": True,
                                                        "delete_option": "Delete"}]},
        })  # fmt: skip  # not waited for: the controller polls its power state
        return {"id": name}

    def machine(self, machine_id: str) -> Machine:
        compute = self.clients["compute"]
        try:
            view = compute.virtual_machines.instance_view(self.settings.resource_group,
                                                          machine_id)  # fmt: skip
        except Exception as exc:
            if "NotFound" in type(exc).__name__ or "ResourceNotFound" in str(exc):
                return Machine(machine_id, machine_id, GONE)
            raise
        codes = [str(getattr(st, "code", "")) for st in (getattr(view, "statuses", None) or [])]
        if any(c == "ProvisioningState/failed" or c.startswith("ProvisioningState/failed")
               for c in codes):  # fmt: skip
            return Machine(machine_id, machine_id, FAILED, ", ".join(codes))
        power = next((c for c in codes if c.startswith("PowerState/")), None)
        state = POWER.get(power or "", STARTING)  # no power state yet: still being created
        return Machine(machine_id, machine_id, state, power or "creating")

    def delete(self, machine_id: str) -> None:
        try:  # the NIC, public IP and disk go with it (delete_option)
            self.clients["compute"].virtual_machines.begin_delete(self.settings.resource_group,
                                                                  machine_id)  # fmt: skip
        except Exception as exc:
            if "NotFound" not in type(exc).__name__ and "ResourceNotFound" not in str(exc):
                raise

    def ours(self, prefix: str) -> list[Machine]:
        prefix = vm_name(prefix)
        vms = self.clients["compute"].virtual_machines.list(self.settings.resource_group)
        return [self.machine(str(vm.name)) for vm in vms if str(vm.name).startswith(prefix)]

    def logs(self, machine_id: str) -> str | None:
        return None  # boot diagnostics are not set up; the job's own log is in storage

    def price_usd_per_h(self) -> float | None:
        query = (f"armRegionName eq '{self.settings.location}' and armSkuName eq "
                 f"'{self.settings.vm_size}' and priceType eq 'Consumption'")  # fmt: skip
        url = f"{PRICES}?$filter={urllib.parse.quote(query)}"
        status, raw = self.transport("GET", url, {"Accept": "application/json"}, None, 30.0)
        if status != 200:
            return None
        for item in json.loads(raw).get("Items", []):
            text = f"{item.get('productName', '')} {item.get('skuName', '')}"
            if any(w in text for w in ("Windows", "Spot", "Low Priority")):
                continue
            if item.get("retailPrice"):
                return float(item["retailPrice"])
        return None

    def balance_usd(self) -> float | None:
        return None  # no credit API: estimated from the credit you entered
