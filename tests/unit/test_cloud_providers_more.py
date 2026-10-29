"""The hyperscaler clients beyond starting a machine (plan G5): states, GPU Relay's own
machines, logs, prices and credit, with fakes shaped like each SDK's documented answers."""

from __future__ import annotations

import base64
import json
from typing import Any

from gpurelay.backends.cloud import FAILED, GONE, RUNNING, STARTING, STOPPED
from gpurelay.backends.providers import make_provider
from gpurelay.config import AwsSettings, AzureSettings, GcpSettings

OWNER = "ab12cd34"
PREFIX = f"gpurelay-{OWNER}-"


class Paginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.query: dict[str, Any] = {}

    def paginate(self, **query: Any) -> list[dict[str, Any]]:
        self.query = query
        return self.pages


def _instance(iid: str, name: str, state: str) -> dict[str, Any]:
    tags = [{"Key": "Name", "Value": name}, {"Key": "gpurelay", "Value": OWNER}]
    return {"InstanceId": iid, "State": {"Name": state}, "Tags": tags}


class Aws:
    def __init__(self) -> None:
        self.pages = Paginator([{"Reservations": [{"Instances": [
            _instance("i-1", PREFIX + "j-0", "running"),
            _instance("i-2", PREFIX + "j-1", "stopped"),
            _instance("i-3", "someone-elses", "running"),
        ]}]}])  # fmt: skip
        self.not_found = False

    def client(self, service: str, region_name: str | None = None) -> Any:
        outer = self

        class Client:
            def get_paginator(self, name: str) -> Paginator:
                if outer.not_found:
                    raise RuntimeError("An error occurred (InvalidInstanceID.NotFound)")
                return outer.pages

            def get_console_output(self, InstanceId: str, Latest: bool) -> dict[str, str]:
                return {"Output": base64.b64encode(b"cloud-init: bootstrap ran").decode()}

            def get_products(self, **kw: Any) -> dict[str, Any]:
                outer.price_query = kw
                product = {"terms": {"OnDemand": {"x": {"priceDimensions": {
                    "y": {"pricePerUnit": {"USD": "0.5260000000"}}}}}}}  # fmt: skip
                return {"PriceList": [json.dumps(product)]}

            def get_caller_identity(self) -> dict[str, str]:
                return {"Account": "123456789012"}

            def get_credits(self, accountId: str, startDate: Any) -> dict[str, Any]:
                outer.credit_account = accountId
                return {"credits": [
                    {"remainingAmount": {"currencyAmount": "75.50"}, "creditStatus": "ACTIVE"},
                    {"remainingAmount": {"currencyAmount": "10.00"}, "creditStatus": "EXPIRED"},
                ]}  # fmt: skip

        return Client()


def _aws(fake: Aws) -> Any:
    keys = {"access_key_id": "AKIA", "secret_access_key": "s"}
    return make_provider("aws", AwsSettings(), keys.get, session_factory=lambda *a: fake)


def test_aws_states_own_machines_logs_price_and_credit() -> None:
    fake = Aws()
    p = _aws(fake)
    assert p.machine("i-1").state == RUNNING and p.machine("i-2").state == STOPPED
    mine = p.ours(PREFIX)
    assert [(m.id, m.state) for m in mine] == [("i-1", RUNNING), ("i-2", STOPPED)]
    filters = {f["Name"]: f["Values"] for f in fake.pages.query["Filters"]}
    assert filters["tag:gpurelay"] == [OWNER] and "terminated" not in filters["instance-state-name"]
    assert p.logs("i-1") == "cloud-init: bootstrap ran"
    assert p.price_usd_per_h() == 0.526
    wanted = {f["Field"]: f["Value"] for f in fake.price_query["Filters"]}
    assert wanted["instanceType"] == "g4dn.xlarge" and wanted["regionCode"] == "us-east-1"
    assert p.balance_usd() == 75.50 and fake.credit_account == "123456789012"  # active only
    fake.not_found = True
    assert p.machine("i-9").state == GONE


def test_aws_and_azure_need_their_keys() -> None:
    assert "AWS access key" in (make_provider("aws", AwsSettings(), {}.get).missing() or "")
    assert "service principal" in (make_provider("azure", AzureSettings(), {}.get).missing() or "")


class Status:
    def __init__(self, code: str) -> None:
        self.code = code


class Vm:
    def __init__(self, name: str) -> None:
        self.name = name


class NotFoundError(Exception):
    pass


def _azure(views: dict[str, Any], prices: dict[str, Any] | None = None) -> Any:
    deleted: list[str] = []

    class Vms:
        def instance_view(self, rg: str, name: str) -> Any:
            if name not in views:
                raise NotFoundError("ResourceNotFound")
            return type("View", (), {"statuses": [Status(c) for c in views[name]]})()

        def list(self, rg: str) -> list[Vm]:
            return [Vm(n) for n in views]

        def begin_delete(self, rg: str, name: str) -> None:
            if name not in views:
                raise NotFoundError("ResourceNotFound")
            deleted.append(name)

    clients = {"compute": type("C", (), {"virtual_machines": Vms()})()}

    def transport(method: str, url: str, headers: dict[str, str], body: bytes | None,
                  timeout: float) -> tuple[int, bytes]:  # fmt: skip
        return 200, json.dumps(prices or {"Items": []}).encode()

    keys = {"tenant_id": "t", "client_id": "c", "client_secret": "x", "subscription_id": "s"}
    p = make_provider("azure", AzureSettings(), keys.get, clients_factory=lambda *a: clients,
                      transport=transport)  # fmt: skip
    p.deleted = deleted
    return p


def test_azure_power_states_own_machines_and_retail_price() -> None:
    views = {
        PREFIX + "j-0": ["ProvisioningState/succeeded", "PowerState/running"],
        PREFIX + "j-1": ["ProvisioningState/succeeded", "PowerState/stopped"],  # still bills
        PREFIX + "j-2": ["ProvisioningState/creating"],
        PREFIX + "j-3": ["ProvisioningState/failed/AllocationFailed"],
        "my-own-vm": ["PowerState/running"],
    }
    prices = {"Items": [
        {"productName": "Virtual Machines NCasT4_v3 Series Windows", "skuName": "NC4as T4 v3",
         "retailPrice": 0.71},
        {"productName": "Virtual Machines NCasT4_v3 Series", "skuName": "NC4as T4 v3 Spot",
         "retailPrice": 0.12},
        {"productName": "Virtual Machines NCasT4_v3 Series", "skuName": "NC4as T4 v3",
         "retailPrice": 0.526},
    ]}  # fmt: skip
    p = _azure(views, prices)
    states = {n: p.machine(n).state for n in views}
    assert states[PREFIX + "j-0"] == RUNNING and states[PREFIX + "j-1"] == STOPPED
    assert states[PREFIX + "j-2"] == STARTING and states[PREFIX + "j-3"] == FAILED
    assert p.machine("gone-vm").state == GONE
    assert sorted(m.id for m in p.ours(PREFIX)) == [PREFIX + f"j-{i}" for i in range(4)]
    assert p.price_usd_per_h() == 0.526  # Linux, pay as you go: not Windows, not Spot
    p.delete(PREFIX + "j-0")
    p.delete("gone-vm")  # already gone: fine
    assert p.deleted == [PREFIX + "j-0"]


def test_google_cloud_lists_its_labelled_vms_and_reads_the_serial_port() -> None:
    calls: list[str] = []

    def transport(method: str, url: str, headers: dict[str, str], body: bytes | None,
                  timeout: float) -> tuple[int, bytes]:  # fmt: skip
        calls.append(url)
        if "filter=labels.gpurelay" in url:
            items = [{"name": PREFIX + "j-0", "status": "RUNNING"},
                     {"name": "other-vm", "status": "RUNNING"}]  # fmt: skip
            return 200, json.dumps({"items": items}).encode()
        if url.endswith("serialPort?port=1"):
            return 200, json.dumps({"contents": "startup-script: bootstrap ran"}).encode()
        if method == "GET":
            return 404, b'{"error": {"code": 404}}'
        return 200, b"{}"

    key = json.dumps({"type": "service_account", "project_id": "proj"})
    p = make_provider("gcp", GcpSettings(), {"service_account_key": key}.get, transport=transport,
                      token_source=lambda info: ("tok", 9e12))  # fmt: skip
    assert [m.id for m in p.ours(PREFIX)] == [PREFIX + "j-0"]
    assert f"filter=labels.gpurelay%3D{OWNER}" in calls[0]
    assert p.logs(PREFIX + "j-0") == "startup-script: bootstrap ran"
    assert p.machine("gone-vm").state == GONE
    assert p.price_usd_per_h() == 0.54 and p.balance_usd() is None
    custom = make_provider("gcp", GcpSettings(machine_type="n1-highmem-8"),
                           {"service_account_key": key}.get)  # fmt: skip
    assert custom.price_usd_per_h() is None  # unknown: the user sets usd_per_hour
