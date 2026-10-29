"""AWS (plan G5): one EC2 GPU instance per segment through boto3 (the ``aws`` extra).
Checked against docs.aws.amazon.com on 2026-10-29 (``docs/verify.md`` #37); not tried on a
real account.

* ``run_instances`` with the Deep Learning Base AMI (``ami_id``, or the latest from the SSM
  parameter ``ami_parameter``; VERIFY: its docs page could not be read), ``instance_type``,
  the session script as ``UserData`` (at most 16 KB; boto3 base64-encodes it; it holds the
  storage token, which only the account can read), tags ``Name`` and ``gpurelay=<owner>``,
  and ``InstanceInitiatedShutdownBehavior="terminate"``: the session's ``shutdown -h now``
  terminates the instance, which ends its billing.
* ``describe_instances`` ``State.Name``: pending | running | shutting-down | terminated |
  stopping | stopped. ``terminate_instances``.
* New accounts have a quota of 0 vCPUs for "Running On-Demand G and VT instances"
  (L-DB2E81BA): request more in Service Quotas first.
* Price: the Price List API (``pricing.get_products``, us-east-1), On-Demand Linux.
  Credits: ``billing.get_credits`` (``remainingAmount`` of each credit).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from gpurelay.backends.cloud import FAILED, GONE, RUNNING, STARTING, STOPPED, CloudError, Machine
from gpurelay.config import AwsSettings

USER_DATA_MAX = 16 * 1024
STATES = {"pending": STARTING, "running": RUNNING, "stopping": STOPPED, "stopped": STOPPED,
          "shutting-down": GONE, "terminated": GONE}  # fmt: skip


def boto3_session(key_id: str, secret: str, region: str) -> Any:
    import boto3  # optional extra: gpurelay[aws]

    return boto3.session.Session(aws_access_key_id=key_id, aws_secret_access_key=secret,
                                 region_name=region)  # fmt: skip


@dataclass
class AwsProvider:
    settings: AwsSettings
    secret: Callable[[str], str | None]
    session_factory: Callable[[str, str, str], Any] = boto3_session
    label: str = "AWS"
    vm: bool = True
    self_end: str = "shutdown -h now"  # terminates (InstanceInitiatedShutdownBehavior)
    _session: Any = field(default=None, init=False)

    @property
    def gpu_label(self) -> str:
        return self.settings.instance_type

    def missing(self) -> str | None:
        if not (self.secret("access_key_id") and self.secret("secret_access_key")):
            return "add your AWS access key on Your GPU platforms"
        import importlib.util

        if self.session_factory is boto3_session and importlib.util.find_spec("boto3") is None:
            return "pip install 'gpurelay[aws]'"
        return None

    def _client(self, service: str, region: str | None = None) -> Any:
        if self._session is None:
            key, secret = self.secret("access_key_id"), self.secret("secret_access_key")
            if not key or not secret:
                raise CloudError("no AWS access key")
            self._session = self.session_factory(key, secret, self.settings.region)
        return self._session.client(service, region_name=region or self.settings.region)

    def _ami(self) -> str:
        if self.settings.ami_id:
            return self.settings.ami_id
        answer = self._client("ssm").get_parameter(Name=self.settings.ami_parameter)
        return str(answer["Parameter"]["Value"])

    def start(self, name: str, script: str, env: dict[str, str]) -> dict[str, str]:
        if len(script.encode("utf-8")) > USER_DATA_MAX:
            raise CloudError("the session script is over EC2's 16 KB user data limit")
        ec2 = self._client("ec2")
        ami = self._ami()
        images = ec2.describe_images(ImageIds=[ami]).get("Images") or []
        root = images[0].get("RootDeviceName", "/dev/sda1") if images else "/dev/sda1"
        owner = name.split("-")[1]
        answer = ec2.run_instances(
            ImageId=ami, InstanceType=self.settings.instance_type, MinCount=1, MaxCount=1,
            UserData=script,  # boto3 base64-encodes it
            InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{"DeviceName": root, "Ebs": {
                "VolumeSize": self.settings.disk_gb, "VolumeType": "gp3",
                "DeleteOnTermination": True}}],
            TagSpecifications=[{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": name}, {"Key": "gpurelay", "Value": owner}]}],
        )  # fmt: skip
        instances = answer.get("Instances") or []
        if not instances:
            raise CloudError(f"EC2 did not start an instance: {str(answer)[:200]}")
        return {"id": str(instances[0]["InstanceId"]), "ami": ami}

    def _machines(self, **query: Any) -> list[Machine]:
        found = []
        for page in self._client("ec2").get_paginator("describe_instances").paginate(**query):
            for reservation in page.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                    state = str((inst.get("State") or {}).get("Name", ""))
                    found.append(Machine(str(inst["InstanceId"]), tags.get("Name", ""),
                                         STATES.get(state, FAILED), state))  # fmt: skip
        return found

    def machine(self, machine_id: str) -> Machine:
        try:
            found = self._machines(InstanceIds=[machine_id])
        except Exception as exc:
            if "InvalidInstanceID.NotFound" in str(exc):
                return Machine(machine_id, "", GONE)
            raise
        return found[0] if found else Machine(machine_id, "", GONE)

    def delete(self, machine_id: str) -> None:
        try:
            self._client("ec2").terminate_instances(InstanceIds=[machine_id])
        except Exception as exc:
            if "InvalidInstanceID.NotFound" not in str(exc):
                raise

    def ours(self, prefix: str) -> list[Machine]:
        owner = prefix.split("-")[1]
        alive = ["pending", "running", "stopping", "stopped"]
        filters = [{"Name": "tag:gpurelay", "Values": [owner]},
                   {"Name": "instance-state-name", "Values": alive}]  # fmt: skip
        return [m for m in self._machines(Filters=filters) if m.name.startswith(prefix)]

    def logs(self, machine_id: str) -> str | None:
        output = self._client("ec2").get_console_output(InstanceId=machine_id, Latest=True)
        text = str(output.get("Output") or "")
        try:
            return base64.b64decode(text, validate=True).decode("utf-8", "replace")
        except ValueError:
            return text or None  # already decoded

    def price_usd_per_h(self) -> float | None:
        """On-Demand Linux price from the Price List API (VERIFY: filter field names)."""
        pricing = self._client("pricing", "us-east-1")
        wanted = [
            ("instanceType", self.settings.instance_type),
            ("regionCode", self.settings.region),
            ("operatingSystem", "Linux"),
            ("tenancy", "Shared"),
            ("preInstalledSw", "NA"),
            ("capacitystatus", "Used"),
        ]
        filters = [{"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in wanted]
        answer = pricing.get_products(ServiceCode="AmazonEC2", MaxResults=10, Filters=filters)
        for raw in answer.get("PriceList", []):
            product = json.loads(raw) if isinstance(raw, str) else raw
            for term in (product.get("terms", {}).get("OnDemand") or {}).values():
                for dimension in (term.get("priceDimensions") or {}).values():
                    usd = (dimension.get("pricePerUnit") or {}).get("USD")
                    if usd and float(usd) > 0:
                        return float(usd)
        return None

    def balance_usd(self) -> float | None:
        """Remaining promotional credit (``billing.get_credits``); None when there is none."""
        account = self._client("sts").get_caller_identity()["Account"]
        since = datetime.now(timezone.utc) - timedelta(days=360)
        answer = self._client("billing", "us-east-1").get_credits(accountId=account,
                                                                  startDate=since)  # fmt: skip
        total, found = 0.0, False
        for credit in answer.get("credits", []):
            amount = (credit.get("remainingAmount") or {}).get("currencyAmount")
            if amount is not None and str(credit.get("creditStatus", "ACTIVE")).upper() in (
                "ACTIVE",
                "AVAILABLE",
            ):  # VERIFY: the status values
                total, found = total + float(amount), True
        return total if found else None
