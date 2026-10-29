"""The user's paid GPU platforms as backends (spec §21, plan G3-G5), with fakes: the session
script, one machine per segment that nothing leaves billing, each platform's requests as its
documentation describes them, and turning a platform on from Your GPU platforms."""

from __future__ import annotations

import base64
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from test_controller import _harness

from gpurelay.backends.cloud import (
    BOOT_LIMIT_S,
    GONE,
    RUNNING,
    STARTING,
    STOPPED,
    CloudBackend,
    CloudError,
    Machine,
    owner_tag,
    session_script,
)
from gpurelay.backends.providers import make_provider
from gpurelay.config import (
    AwsSettings,
    AzureSettings,
    Config,
    DigitalOceanSettings,
    GcpSettings,
    LambdaSettings,
    RunPodSettings,
    RuntimeSettings,
    StorageSettings,
    VastSettings,
)
from gpurelay.controller.registry import build_backends
from gpurelay.models import BackendRunState, Job, JobKind, JobState, LaunchHandle, Segment
from gpurelay.storage import LocalFSStorage

TOKEN = "hf_" + "t" * 30
STORE = "hf:ada/gpurelay-store"


def _unpack(script: str) -> str:
    packed = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", script)
    assert packed
    return zlib.decompress(base64.b64decode(packed.group(1))).decode()


def test_the_session_script_runs_the_bootstrap_under_a_hard_limit() -> None:
    container = session_script(name="gpurelay-x-j-0", bootstrap="print('hi')\n", hard_limit_s=3600,
                               vm=False, self_end="echo bye")  # fmt: skip
    assert _unpack(container) == "print('hi')\n"
    assert "timeout -k 120 3600" in container and container.rstrip().endswith("echo bye")
    assert TOKEN not in container  # containers get the token from the platform
    vm = session_script(name="n", bootstrap="x=1", hard_limit_s=60, vm=True, self_end="",
                        env={"HF_TOKEN": TOKEN})  # fmt: skip
    assert f"export HF_TOKEN='{TOKEN}'" in vm and "python3 -m venv" in vm


@dataclass
class FakeProvider:
    label: str = "Fake"
    gpu_label: str = "RTX 4090"
    vm: bool = False
    self_end: str = ""
    machines: dict[str, Machine] = field(default_factory=dict)
    started: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    price: float | None = 0.69
    price_calls: int = 0

    def missing(self) -> str | None:
        return None

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        machine_id = f"m{len(self.started)}"
        self.started.append((name, script, env))
        self.machines[machine_id] = Machine(machine_id, name, RUNNING)
        return {"id": machine_id, "usd_per_h": "0.74"}

    def machine(self, machine_id: str) -> Machine:
        return self.machines.get(machine_id, Machine(machine_id, "", GONE))

    def delete(self, machine_id: str) -> None:
        self.deleted.append(machine_id)
        if machine_id in self.machines:
            self.machines[machine_id].state = GONE

    def ours(self, prefix: str) -> list[Machine]:
        return [m for m in self.machines.values() if m.name.startswith(prefix)]

    def logs(self, machine_id: str) -> str | None:
        return "log"

    def price_usd_per_h(self) -> float | None:
        self.price_calls += 1
        return self.price

    def balance_usd(self) -> float | None:
        return 12.0


def _backend(provider: FakeProvider, clock: Any = None, **settings: Any) -> CloudBackend:
    return CloudBackend(
        name="runpod", settings=RunPodSettings(**settings), runtime=RuntimeSettings(),
        storage_uri=STORE, wheel_for=lambda job_id: "gpurelay-1.1.0-py3-none-any.whl",
        provider=provider, token_source=lambda: TOKEN,
        **({"clock": clock} if clock else {}),
    )  # fmt: skip


def _job_segment() -> tuple[Job, Segment]:
    job = Job(name="vae", kind=JobKind.SCRIPT, entry="vae.py")
    return job, Segment(job_id=job.id, index=2, backend="runpod")


def test_one_machine_per_segment_named_for_this_gpu_relay() -> None:
    provider = FakeProvider()
    backend = _backend(provider)
    job, seg = _job_segment()
    handle = backend.launch(job, seg, deadline_s=3600)
    [(name, script, env)] = provider.started
    assert name == f"gpurelay-{owner_tag(STORE)}-{job.id}-2".lower() and env == {"HF_TOKEN": TOKEN}
    assert TOKEN not in script and handle.extra["usd_per_h"] == "0.74"
    assert backend.poll(handle) is BackendRunState.RUNNING
    provider.machines[handle.handle_id].state = STOPPED
    assert backend.poll(handle) is BackendRunState.FINISHED  # status.json says how it ended
    backend.cancel(handle)
    assert provider.deleted == [handle.handle_id]


def test_a_machine_that_never_finishes_starting_is_given_up() -> None:
    now = [1000.0]
    provider = FakeProvider()
    backend = _backend(provider, clock=lambda: now[0])
    handle = backend.launch(*_job_segment(), deadline_s=3600)
    provider.machines[handle.handle_id].state = STARTING
    assert backend.poll(handle) is BackendRunState.QUEUED
    now[0] += BOOT_LIMIT_S + 1
    assert backend.poll(handle) is BackendRunState.ERROR


def test_the_sweep_deletes_only_this_gpu_relays_machines_that_no_segment_owns() -> None:
    provider = FakeProvider()
    backend = _backend(provider)
    owner = owner_tag(STORE)
    provider.machines = {
        "live": Machine("live", f"gpurelay-{owner}-j1-0", RUNNING),
        "orphan": Machine("orphan", f"gpurelay-{owner}-j2-3", RUNNING),
        "other": Machine("other", "gpurelay-ffffffff-j9-0", RUNNING),  # another GPU Relay's
        "mine": Machine("mine", "my-notebook", RUNNING),  # the user's own machine
    }
    assert backend.sweep({"live"}) == ["orphan"]
    assert provider.deleted == ["orphan"]


def test_the_price_comes_from_settings_or_the_platform_hourly() -> None:
    now = [0.0]
    provider = FakeProvider()
    backend = _backend(provider, clock=lambda: now[0])
    assert backend.price_usd_per_h() == 0.69 and provider.price_calls == 1
    now[0] += 60
    assert backend.price_usd_per_h() == 0.69 and provider.price_calls == 1
    now[0] += 3600
    backend.price_usd_per_h()
    assert provider.price_calls == 2
    assert _backend(FakeProvider(), usd_per_hour=0.5).price_usd_per_h() == 0.5


def test_the_controller_deletes_a_machine_as_soon_as_its_segment_is_over(tmp_path: Path) -> None:
    h = _harness(tmp_path, {})
    provider = FakeProvider()
    backend = _backend(provider, clock=h.clock)
    backend.storage_uri = STORE
    from gpurelay.controller.quota import Budget
    from gpurelay.controller.registry import BackendSlot
    from gpurelay.models import BudgetModel

    h.controller.backends["runpod"] = BackendSlot(
        name="runpod", backend=backend, budget=Budget(BudgetModel.MONTHLY, 0.0, estimate=False),
        automated=True, max_concurrent=1, max_session_s=6 * 3600, gpu_label="RTX 4090",
        paid_usd_per_h=None, credit_usd=20.0, credit_set_at=h.clock.now, paid_only=True,
    )  # fmt: skip
    queued = h.submit("train")
    h.controller.tick()
    [seg] = h.db.segments(queued.id)
    assert seg.paid_from == "credit" and seg.paid_usd_per_h == 0.74  # what the machine costs
    # The runtime finished, but the machine is still up (Lambda-like): finish and delete it.
    from gpurelay.models import RuntimeState, StatusFile, utcnow
    from gpurelay.storage.layout import status_key

    status = StatusFile(job_id=queued.id, segment_id=seg.id, state=RuntimeState.HANDOFF,
                        heartbeat_at=utcnow(), message="handoff")  # fmt: skip
    h.storage.put_bytes(status.model_dump_json().encode(), status_key(queued.id))
    h.controller.tick()
    [seg] = h.db.segments(queued.id)[:1]
    assert seg.outcome == "handoff" and seg.handle_id in provider.deleted
    # A machine nobody tracks (a crash between start and save) is removed by the sweep.
    owner = owner_tag(STORE)
    provider.machines["stray"] = Machine("stray", f"gpurelay-{owner}-zzz-0", RUNNING)
    h.controller._swept_at = -1e18
    h.controller.tick()
    assert "stray" in provider.deleted
    live = {s.handle_id for s in h.db.active_segments()}
    assert not live & set(provider.deleted)  # the job's next machine was kept


# ---------------------------------------------------------------- the platforms' requests


@dataclass
class Http:
    """A fake HTTPS endpoint: answers by (method, path prefix), records every call."""

    answers: dict[tuple[str, str], Any]
    calls: list[tuple[str, str, Any, dict[str, str]]] = field(default_factory=list)

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None,
                 timeout: float) -> tuple[int, bytes]:  # fmt: skip
        data = json.loads(body) if body else None
        self.calls.append((method, url, data, headers))
        for (m, prefix), answer in self.answers.items():
            if m == method and prefix in url:
                status, payload = answer if isinstance(answer, tuple) else (200, answer)
                return status, json.dumps(payload).encode() if payload is not None else b""
        return 404, b'{"error": "not found"}'

    def sent(self, method: str, part: str) -> Any:
        return next(d for m, u, d, _h in self.calls if m == method and part in u)


def _secret(values: dict[str, str]) -> Any:
    return lambda key: values.get(key)


def test_runpod_uses_api_v2_with_the_token_in_the_pod_environment() -> None:
    http = Http({
        ("POST", "/v2/pods"): {"id": "pod1", "cost": 0.69},
        ("GET", "/v2/pods/pod1"): {"id": "pod1", "name": "n", "status": "EXITED"},
        ("GET", "/v2/catalog/gpus"): {"gpus": [{"id": "NVIDIA GeForce RTX 4090",
                                                "price": {"secure": 0.69, "community": 0.34}}]},
        ("POST", "/graphql"): {"data": {"myself": {"clientBalance": 41.5}}},
        ("DELETE", "/v2/pods/pod1"): (204, None),
    })  # fmt: skip
    p = make_provider("runpod", RunPodSettings(), _secret({"api_key": "rpa_key"}), transport=http)
    assert p.start("gpurelay-x-j-0", "#!/bin/bash\necho", {"HF_TOKEN": TOKEN}) == {
        "id": "pod1", "usd_per_h": "0.69"}  # fmt: skip
    body = http.sent("POST", "/v2/pods")
    assert body["env"] == {"HF_TOKEN": TOKEN} and body["gpu"] == {"id": "NVIDIA GeForce RTX 4090",
                                                                  "count": 1}  # fmt: skip
    assert body["entrypoint"] == ["/bin/bash", "-c"] and TOKEN not in body["cmd"][0]
    assert http.calls[0][3]["Authorization"] == "Bearer rpa_key"
    assert p.machine("pod1").state == STOPPED and p.machine("nope").state == GONE
    assert p.price_usd_per_h() == 0.69 and p.balance_usd() == 41.5
    p.delete("pod1")
    assert p.missing() is None
    assert make_provider("runpod", RunPodSettings(), _secret({})).missing()


def test_vast_rents_the_cheapest_verified_offer_with_a_short_onstart() -> None:
    http = Http({
        ("POST", "/bundles/"): {"offers": [{"id": 77, "gpu_name": "RTX 4090", "dph_total": 0.31}]},
        ("PUT", "/asks/77/"): {"success": True, "new_contract": 555},
        ("GET", "/instances/555/"): {"instances": {"id": 555, "label": "n", "actual_status": None}},
        ("GET", "/users/current/"): {"credit": 9.25},
    })  # fmt: skip
    p = make_provider("vastai", VastSettings(), _secret({"api_key": "vk"}), transport=http)
    assert p.start("gpurelay-x-j-0", "echo hi", {"HF_TOKEN": TOKEN}) == {
        "id": "555", "offer": "77", "usd_per_h": "0.31"}  # fmt: skip
    query = http.sent("POST", "/bundles/")
    assert query["gpu_name"] == {"eq": "RTX 4090"} and query["verified"] == {"eq": True}
    assert query["order"] == [["dph_total", "asc"]]
    rent = http.sent("PUT", "/asks/77/")
    assert rent["env"] == f"-e HF_TOKEN={TOKEN}" and rent["onstart"] == "echo hi"
    assert p.machine("555").state == STARTING and p.balance_usd() == 9.25
    with pytest.raises(CloudError, match="characters"):
        p.start("n", "x" * 5000, {"HF_TOKEN": TOKEN})


def test_lambda_adds_its_ssh_key_once_picks_a_region_with_capacity_and_terminates() -> None:
    types = {"data": {"gpu_1x_a10": {
        "instance_type": {"name": "gpu_1x_a10", "price_cents_per_hour": 75},
        "regions_with_capacity_available": [{"name": "us-east-1"}]}}}  # fmt: skip
    http = Http({
        ("GET", "/instance-types"): types,
        ("GET", "/ssh-keys"): {"data": []},
        ("POST", "/ssh-keys"): {"data": {"id": "k", "name": "gpurelay", "private_key": "x"}},
        ("POST", "/instance-operations/launch"): {"data": {"instance_ids": ["i-1"]}},
        ("GET", "/instances/i-1"): {"data": {"id": "i-1", "name": "n", "status": "active"}},
        ("POST", "/instance-operations/terminate"): {"data": {"terminated_instances": []}},
    })  # fmt: skip
    p = make_provider("lambda", LambdaSettings(), _secret({"api_key": "lk"}), transport=http)
    started = p.start("gpurelay-x-j-0", "#!/bin/bash\nrun", {"HF_TOKEN": TOKEN})
    assert started == {"id": "i-1", "region": "us-east-1", "usd_per_h": "0.75"}
    launch = http.sent("POST", "/instance-operations/launch")
    assert launch["ssh_key_names"] == ["gpurelay"] and launch["user_data"] == "#!/bin/bash\nrun"
    assert launch["region_name"] == "us-east-1"
    assert p.machine("i-1").state == RUNNING
    p.delete("i-1")
    assert http.sent("POST", "/instance-operations/terminate") == {"instance_ids": ["i-1"]}
    assert p.self_end == "" and p.balance_usd() is None  # shutdown keeps billing; no balance API


def test_digitalocean_creates_a_tagged_gpu_droplet_and_reads_its_price() -> None:
    http = Http({
        ("POST", "/droplets"): {"droplet": {"id": 4242}},
        ("GET", "/droplets/4242"): {"droplet": {"id": 4242, "name": "n", "status": "off"}},
        ("GET", "/sizes"): {"sizes": [{"slug": "gpu-4000adax1-20gb", "price_hourly": 0.76}]},
        ("DELETE", "/droplets/4242"): (204, None),
    })  # fmt: skip
    p = make_provider("digitalocean", DigitalOceanSettings(), _secret({"token": "dop"}),
                      transport=http)  # fmt: skip
    assert p.start("gpurelay-x-job_1-0", "#!/bin/bash", {}) == {"id": "4242"}
    body = http.sent("POST", "/droplets")
    assert body["image"] == "gpu-h100x1-base" and body["region"] == "tor1"
    assert body["name"] == "gpurelay-x-job-1-0" and body["tags"] == ["gpurelay"]
    assert p.machine("4242").state == STOPPED  # powered off still bills: then deleted
    assert p.price_usd_per_h() == 0.76


def test_google_cloud_inserts_a_gpu_vm_with_the_startup_script() -> None:
    http = Http({
        ("POST", "/instances"): {"kind": "compute#operation"},
        ("GET", "/instances/gpurelay-ab12cd34-j-0"): {"name": "gpurelay-ab12cd34-j-0",
                                                      "status": "TERMINATED"},
    })  # fmt: skip
    key = json.dumps({"type": "service_account", "project_id": "proj"})
    p = make_provider("gcp", GcpSettings(), _secret({"service_account_key": key}),
                      transport=http, token_source=lambda info: ("ya29.token", 9e12))  # fmt: skip
    assert p.start("gpurelay-ab12cd34-j-0", "#!/bin/bash\nrun", {}) == {
        "id": "gpurelay-ab12cd34-j-0"}  # fmt: skip
    _, url, body, headers = http.calls[0]
    assert "/projects/proj/zones/us-central1-a/instances" in url
    assert headers["Authorization"] == "Bearer ya29.token"
    assert body["scheduling"]["onHostMaintenance"] == "TERMINATE"
    assert body["guestAccelerators"][0]["acceleratorType"].endswith("nvidia-tesla-t4")
    items = {i["key"]: i["value"] for i in body["metadata"]["items"]}
    assert items == {"startup-script": "#!/bin/bash\nrun", "install-nvidia-driver": "True"}
    assert body["labels"] == {"gpurelay": "ab12cd34"}
    assert p.machine("gpurelay-ab12cd34-j-0").state == STOPPED


class FakeEc2:
    def __init__(self) -> None:
        self.run: dict[str, Any] = {}
        self.terminated: list[str] = []

    def describe_images(self, ImageIds: list[str]) -> dict[str, Any]:
        return {"Images": [{"RootDeviceName": "/dev/sda1"}]}

    def run_instances(self, **kw: Any) -> dict[str, Any]:
        self.run = kw
        return {"Instances": [{"InstanceId": "i-0abc"}]}

    def terminate_instances(self, InstanceIds: list[str]) -> None:
        self.terminated += InstanceIds

    def get_parameter(self, Name: str) -> dict[str, Any]:
        return {"Parameter": {"Value": "ami-123"}}


def test_aws_terminates_itself_on_shutdown_and_tags_its_instances() -> None:
    ec2 = FakeEc2()

    class Session:
        def client(self, service: str, region_name: str | None = None) -> Any:
            return ec2

    keys = _secret({"access_key_id": "AKIA", "secret_access_key": "s"})
    p = make_provider("aws", AwsSettings(), keys, session_factory=lambda k, s, r: Session())
    started = p.start("gpurelay-ab12cd34-j-0", "#!/bin/bash\nrun", {})
    assert started == {"id": "i-0abc", "ami": "ami-123"}
    run = ec2.run
    assert run["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert run["UserData"].startswith("#!")
    tags = {t["Key"]: t["Value"] for t in run["TagSpecifications"][0]["Tags"]}
    assert tags == {"Name": "gpurelay-ab12cd34-j-0", "gpurelay": "ab12cd34"}
    assert run["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"] is True
    assert p.self_end == "shutdown -h now"
    p.delete("i-0abc")
    assert ec2.terminated == ["i-0abc"]


def test_azure_puts_ip_nic_and_disk_on_delete_with_the_vm() -> None:
    created: dict[str, Any] = {}

    class Poller:
        def __init__(self, value: Any) -> None:
            self.value = value

        def result(self) -> Any:
            return self.value

    class Res:
        id = "/subscriptions/s/res"

    class Ops:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def begin_create_or_update(self, *args: Any) -> Poller:
            created[self.kind] = args
            return Poller(Res())

        def create_or_update(self, *args: Any) -> None:
            created[self.kind] = args

        def get(self, *args: Any) -> Res:
            return Res()

    class Ns:
        def __init__(self, **ops: Any) -> None:
            self.__dict__.update(ops)

    clients = {
        "resource": Ns(resource_groups=Ops("rg")),
        "network": Ns(virtual_networks=Ops("vnet"), subnets=Ops("subnet"),
                      public_ip_addresses=Ops("ip"), network_interfaces=Ops("nic")),
        "compute": Ns(virtual_machines=Ops("vm")),
    }  # fmt: skip
    values = {"tenant_id": "t", "client_id": "c", "client_secret": "x", "subscription_id": "s"}
    p = make_provider("azure", AzureSettings(), _secret(values), clients_factory=lambda *a: clients,
                      key_source=lambda: "ssh-rsa AAA")  # fmt: skip
    assert p.start("gpurelay-ab12cd34-j-0", "#!/bin/bash\nrun", {}) == {
        "id": "gpurelay-ab12cd34-j-0"}  # fmt: skip
    vm = created["vm"][2]
    assert vm["hardware_profile"] == {"vm_size": "Standard_NC4as_T4_v3"}
    assert vm["storage_profile"]["os_disk"]["delete_option"] == "Delete"
    assert vm["network_profile"]["network_interfaces"][0]["delete_option"] == "Delete"
    assert base64.b64decode(vm["os_profile"]["custom_data"]).decode() == "#!/bin/bash\nrun"
    nic = created["nic"][2]
    assert nic["ip_configurations"][0]["public_ip_address"]["delete_option"] == "Delete"
    assert p.self_end == ""  # a shutdown keeps billing: the controller deletes it


# ---------------------------------------------------------------- built from the config


def test_only_platforms_turned_on_become_backends(tmp_path: Path) -> None:
    config = Config(storage=StorageSettings(kind="localfs", localfs_root=tmp_path / "s"))
    storage = LocalFSStorage(tmp_path / "s")
    assert not {"runpod", "aws"} & set(build_backends(config, storage, tmp_path / "sess"))
    config.backends.runpod.terms_reviewed = True
    config.backends.runpod.enabled = True
    config.backends.runpod.credit_usd = 25.0
    slots = build_backends(config, storage, tmp_path / "sess")
    slot = slots["runpod"]
    assert slot.paid_only and slot.credit_usd == 25.0 and slot.budget.hours == 0.0
    assert not slot.backend.check_ready().ready  # needs Hugging Face storage (and its key)
    with pytest.raises(ValueError, match="terms_reviewed"):
        AwsSettings(enabled=True)


def test_handles_record_where_they_run() -> None:
    handle = LaunchHandle(backend="runpod", handle_id="m0", extra={"started_at": "0"})
    assert handle.extra["started_at"] == "0"
    assert JobState.RUNNING.value == "running"
