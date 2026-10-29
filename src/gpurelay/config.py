"""User configuration: pydantic models plus TOML load/save.

The config file lives at ``$GPURELAY_HOME/config.toml`` (default ``~/.gpurelay``).

Hard rules enforced by the schema (spec §3):

* **One account per service.** Each backend type is a single fixed section; unknown
  sections (e.g. ``[backends.kaggle2]``) are rejected.
* **No secrets in the config file.** Every model forbids unknown keys, and none
  declares a credential field. API keys and tokens come from the OS keyring or
  environment variables.
* **Ungated backends stay off.** Lightning AI cannot be enabled until its terms and
  SDK are verified.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from gpurelay.constants import (
    CONFIG_FILENAME,
    CONFIG_SCHEMA_VERSION,
    DEFAULT_HOME_DIRNAME,
    DEFAULT_KEEP_CHECKPOINTS,
    DEFAULT_MAX_INFRA_RETRIES,
    DEFAULT_MAX_LOST_WORK_S,
    DEFAULT_MIN_CHECKPOINT_INTERVAL_S,
    DEFAULT_MIN_SEGMENT_H,
    DEFAULT_OVERHEAD_BUDGET,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_REPLAY_THRESHOLD_S,
    DEFAULT_SAFETY_MARGIN_S,
    DEFAULT_ZSTD_LEVEL,
    ENV_HOME,
    HEARTBEAT_INTERVAL_S,
    STALE_HEARTBEAT_S,
)
from gpurelay.errors import ConfigError
from gpurelay.models import BudgetModel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_HOUR = 3600


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ====================================================================== sections


class RuntimeSettings(_Section):
    """Checkpointing and in-session behaviour (spec §7, §8, §9, §12)."""

    replay_threshold_s: float = Field(default=DEFAULT_REPLAY_THRESHOLD_S, ge=0)
    overhead_budget: float = Field(default=DEFAULT_OVERHEAD_BUDGET, gt=0, lt=1)
    min_interval_s: float = Field(default=DEFAULT_MIN_CHECKPOINT_INTERVAL_S, gt=0)
    max_lost_work_s: float = Field(default=DEFAULT_MAX_LOST_WORK_S, gt=0)
    safety_margin_s: float = Field(default=DEFAULT_SAFETY_MARGIN_S, ge=0)
    zstd_level: int = Field(default=DEFAULT_ZSTD_LEVEL, ge=1, le=22)
    keep_checkpoints: int = Field(default=DEFAULT_KEEP_CHECKPOINTS, ge=2)
    heartbeat_s: float = Field(default=HEARTBEAT_INTERVAL_S, gt=0)
    stale_heartbeat_s: float = Field(default=STALE_HEARTBEAT_S, gt=0)
    # Automatic instrumentation (spec §7.3, Phase 9): how many levels of nested ``for``
    # loops get safe points (1 = outermost only), and opt-in ``while`` loops and loops
    # inside functions.
    loop_depth: int = Field(default=2, ge=1, le=8)
    instrument_while: bool = False
    instrument_functions: bool = False
    resume_markers: bool = False  # "[gpurelay] resumed on ..." log lines (spec §9)


class SchedulerSettings(_Section):
    """Controller scheduling policy (spec §13.4, §13.5)."""

    min_segment_h: float = Field(default=DEFAULT_MIN_SEGMENT_H, gt=0)
    max_infra_retries: int = Field(default=DEFAULT_MAX_INFRA_RETRIES, ge=0)
    poll_interval_s: int = Field(default=DEFAULT_POLL_INTERVAL_S, gt=0)
    # Extra worker segments per job for relay.chunks(..., parallel=True); 0 disables.
    max_parallel_workers: int = Field(default=2, ge=0)


class StorageSettings(_Section):
    """Durable shared store (spec §11). The access token is never stored here."""

    kind: Literal["hf_hub", "localfs", "gdrive"] = "hf_hub"
    hf_repo_id: str | None = None  # "<user>/gpurelay-store"; always created private
    localfs_root: Path | None = None  # defaults to $GPURELAY_HOME/store
    gdrive_folder_id: str | None = None  # app-created Drive folder (`gpurelay init --gdrive`)


class KaggleSettings(_Section):
    """Kaggle backend (spec §10.4). Automated only via the official API/CLI.

    The Kaggle API key is read from the keyring / ``KAGGLE_KEY``, never from here.
    """

    enabled: bool = False
    username: str | None = None
    # VERIFY: free-tier per-session cap; Kaggle documents 12 h for GPU sessions.
    max_session_s: int = Field(default=12 * _HOUR, gt=0)
    # VERIFY: weekly GPU quota (~30 h/week, varies).
    weekly_gpu_hours: float = Field(default=30.0, ge=0)
    # VERIFY: concurrent GPU sessions allowed per account.
    max_concurrent: int = Field(default=1, ge=1)
    # `machine_shape` of pushed kernels; the official kagglesdk lists "NvidiaTeslaT4",
    # "NvidiaTeslaP100" (and "Tpu1VmV38"). Sent only when enable_gpu is true.
    accelerator: str = "NvidiaTeslaT4"
    # False runs CPU-only sessions (e.g. the smoke test, to save GPU quota).
    enable_gpu: bool = True
    # VERIFY: how the weekly quota resets.
    budget_model: BudgetModel = BudgetModel.WEEKLY_ROLLING
    # "<owner>/<slug>" of the private dataset holding the storage token (spec §10.4).
    secrets_dataset: str | None = None


class ColabSettings(_Section):
    """Colab backend (spec §10.5). Manual only: never automated (hard rule 2)."""

    enabled: bool = False
    # VERIFY: Colab free-tier limits are unpublished; conservative handoff deadline.
    max_session_s: int = Field(default=10 * _HOUR, gt=0)
    budget_model: BudgetModel = BudgetModel.UNKNOWN_OPPORTUNISTIC
    gpu_label: str | None = None  # shown in status; the session reports the real GPU
    # Name of the Colab user secret (key icon) holding the Hugging Face token.
    secret_name: str = "GPURELAY_HF_TOKEN"
    # Ask for Colab before the automated platforms (each session waits for Run all). If
    # nobody presses Run all within ``wait_s``, the job goes to the other platforms.
    first: bool = False
    wait_s: int = Field(default=30 * 60, gt=0)


class LocalSettings(_Section):
    """Local subprocess backend (spec §10.2). No quota."""

    enabled: bool = True
    max_concurrent: int = Field(default=1, ge=1)
    gpu_label: str | None = None  # e.g. "RTX 3060"; None means detect / CPU
    detect_gpu: bool = True  # find an NVIDIA GPU with nvidia-smi when gpu_label is unset
    max_session_s: float = Field(default=7 * 24 * _HOUR, gt=0)


_RESERVED_BACKEND_NAMES = frozenset({
    "local", "kaggle", "colab_manual", "lightning", "modal",
    "runpod", "vastai", "lambda", "digitalocean", "aws", "gcp", "azure",
})  # fmt: skip


class SimulatedSettings(_Section):
    """A simulated backend for testing the scheduler (spec §10.3). Not an account."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    gpu_label: str = "SimGPU"
    max_session_s: float = Field(default=3600.0, gt=0)
    budget_hours: float | None = Field(default=None, ge=0)  # None = unlimited
    budget_model: BudgetModel = BudgetModel.WEEKLY_ROLLING
    # Budget in another unit (P11): budget_hours is then the amount in that unit.
    budget_unit: Literal["h", "usd", "credits"] = "h"
    cost_per_hour: float = Field(default=1.0, gt=0)
    # Paid GPU time after the free budget (tests of the paid order): price and monthly limit.
    paid_usd_per_h: float | None = Field(default=None, gt=0)
    monthly_paid_cap_usd: float = Field(default=0.0, ge=0)
    # Credit the user bought there (spec §21.2), and when they said so.
    credit_usd: float | None = Field(default=None, ge=0)
    credit_set_at: float | None = None
    use_credits: bool = True
    max_concurrent: int = Field(default=1, ge=1)
    kill_after_s: float | None = Field(default=None, gt=0)
    kill_between_s: tuple[float, float] | None = None
    seed: int = 0

    @field_validator("name")
    @classmethod
    def _not_reserved(cls, value: str) -> str:
        if value in _RESERVED_BACKEND_NAMES:
            raise ValueError(f"simulated backend name {value!r} is reserved")
        return value


class LightningSettings(_Section):
    """Lightning AI backend (spec §10.6), through the official ``lightning_sdk`` only.

    Paused (2026-10-25): after a security incident on 15 October 2026 Lightning revoked all
    user access and closed sign-ups while it investigates (lightning.ai/security-incident).
    Keep it off; if it reopens, use a separate fine-grained Hugging Face token for it.

    Gated by hard rule 4: enabling it also requires ``terms_reviewed = true``, i.e. the owner
    read the terms-of-service findings in docs/verify.md (#10) and accepts them.
    """

    enabled: bool = False
    terms_reviewed: bool = False
    # VERIFY: free credits/month and what they buy are not on an official page (docs/verify.md
    # #11): third-party guides say 15 credits/month (~22 T4-hours), the homepage "80 free
    # GPU hours". The conservative figure is the default; correct it with `gpurelay quota set`.
    monthly_gpu_hours: float = Field(default=22.0, ge=0)
    # Or the grant as credits (P11): both set -> hours = monthly_credits / credits_per_hour.
    monthly_credits: float | None = Field(default=None, ge=0)
    credits_per_hour: float | None = Field(default=None, gt=0)
    budget_model: BudgetModel = BudgetModel.MONTHLY
    machine: str = "T4"  # a lightning_sdk ``Machine`` member name (T4, L4, A100, ...)
    # VERIFY: container image the job runs in (CUDA + torch preinstalled).
    image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    teamspace: str | None = None  # "owner/teamspace"; None: the SDK's default teamspace
    interruptible: bool = False  # spot machines: cheaper, may be stopped at any time
    # Lightning sets no session limit (credits do): segments hand off after this long.
    max_session_s: int = Field(default=4 * _HOUR, gt=0)
    max_concurrent: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _gated(self) -> LightningSettings:
        if self.enabled and not self.terms_reviewed:
            raise ValueError(
                "the Lightning AI backend needs `terms_reviewed = true`: read the terms-of-service "
                "findings in docs/verify.md (#10) first (spec §3 rule 4)"
            )
        return self


class ModalSettings(_Section):
    """Modal backend (plan P12), through the official ``modal`` SDK only.

    Gated like Lightning AI (hard rule 4): enabling it also requires ``terms_reviewed = true``
    (the owner read Modal's terms of service and accepts them for this use).
    """

    enabled: bool = False
    terms_reviewed: bool = False
    gpu: str = "T4"  # a Modal GPU name: T4, L4, A10G, A100, ...
    # VERIFY on modal.com/pricing: the Starter plan's monthly credit and the GPU's price.
    monthly_credit_usd: float = Field(default=30.0, ge=0)
    usd_per_hour: float = Field(default=0.59, gt=0)
    # Paid use after the free credit: never more than this a month (0 = free credit only).
    # Raise it only after adding credit to Modal; also set Modal's own spending limit.
    monthly_paid_cap_usd: float = Field(default=0.0, ge=0)
    budget_model: BudgetModel = BudgetModel.MONTHLY
    image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    # One sandbox lives at most 24 h; segments hand off after this long.
    max_session_s: int = Field(default=4 * _HOUR, gt=0, le=24 * _HOUR)
    max_concurrent: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _gated(self) -> ModalSettings:
        if self.enabled and not self.terms_reviewed:
            raise ValueError(
                "the Modal backend needs `terms_reviewed = true`: read Modal's terms of service "
                "first (spec §3 rule 4)"
            )
        return self


class CloudSettings(_Section):
    """A paid GPU platform the user has (spec §21, plan G3-G5): GPU Relay starts one machine
    per segment there, through the platform's official API, and deletes it afterwards.

    Off until the user turns it on and confirms the platform's terms (hard rule 4). Each is
    built from the platform's documentation and fakes; ``docs/verify.md`` #33-39 says what is
    checked and what is not.
    """

    enabled: bool = False
    terms_reviewed: bool = False
    # USD per GPU hour. None: read from the platform's price list (where it has one).
    usd_per_hour: float | None = Field(default=None, gt=0)
    max_session_s: int = Field(default=6 * _HOUR, gt=0, le=24 * _HOUR)
    max_concurrent: int = Field(default=1, ge=1, le=8)
    # Credit the user bought there (spec §21.2) and when they entered it; GPU Relay spends it
    # after free time when ``use_credits``. A platform whose API reports a balance uses that.
    credit_usd: float | None = Field(default=None, ge=0)
    credit_set_at: float | None = None
    use_credits: bool = True
    # Paid by card beyond the credit: never more than this a month (0: credit only).
    monthly_paid_cap_usd: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _gated(self) -> CloudSettings:
        if self.enabled and not self.terms_reviewed:
            raise ValueError(
                f"{type(self).__name__.removesuffix('Settings')} needs `terms_reviewed = true`: "
                "read the platform's terms of service first (spec §3 rule 4)"
            )
        return self


_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


class RunPodSettings(CloudSettings):
    """RunPod Pods through REST API v2 (G3)."""

    gpu: str = "NVIDIA GeForce RTX 4090"  # a GPU type id from GET /v2/catalog/gpus
    cloud: Literal["SECURE", "COMMUNITY"] = "SECURE"
    image: str = _IMAGE
    disk_gb: int = Field(default=40, ge=10, le=1000)


class VastSettings(CloudSettings):
    """Vast.ai instances through its REST API (G3)."""

    gpu: str = "RTX 4090"  # VERIFY: the docs write both "RTX 4090" and "RTX_4090"
    image: str = _IMAGE
    disk_gb: int = Field(default=40, ge=10, le=1000)
    min_reliability: float = Field(default=0.98, ge=0, le=1)
    max_usd_per_hour: float | None = Field(default=None, gt=0)  # skip dearer offers


class LambdaSettings(CloudSettings):
    """Lambda Cloud instances through its Cloud API (G4)."""

    instance_type: str = "gpu_1x_a10"  # a name from GET /instance-types
    region: str | None = None  # None: the first region with capacity


class DigitalOceanSettings(CloudSettings):
    """DigitalOcean GPU Droplets through its API (G4)."""

    size: str = "gpu-4000adax1-20gb"  # RTX 4000 Ada, the cheapest single-GPU plan
    region: str = "tor1"
    image: str = "gpu-h100x1-base"  # the docs: for every single-GPU plan


class AwsSettings(CloudSettings):
    """AWS EC2 GPU instances through boto3 (G5)."""

    instance_type: str = "g4dn.xlarge"  # one NVIDIA T4
    region: str = "us-east-1"
    ami_id: str | None = None  # None: the latest Deep Learning Base AMI (SSM parameter below)
    # VERIFY: the DLAMI docs page that names this parameter could not be read.
    ami_parameter: str = (
        "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
    )
    disk_gb: int = Field(default=100, ge=30, le=2000)


class GcpSettings(CloudSettings):
    """Google Cloud Compute Engine GPU VMs through its REST API (G5)."""

    zone: str = "us-central1-a"
    machine_type: str = "n1-standard-4"
    accelerator: str = "nvidia-tesla-t4"
    image_project: str = "deeplearning-platform-release"
    image_family: str = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
    disk_gb: int = Field(default=100, ge=50, le=2000)


class AzureSettings(CloudSettings):
    """Azure GPU virtual machines through the Azure SDK for Python (G5)."""

    location: str = "eastus"
    vm_size: str = "Standard_NC4as_T4_v3"  # VERIFY: one T4 (NCasT4_v3 family)
    resource_group: str = "gpurelay"  # created if missing; only GPU Relay's machines in it
    image: str = "microsoft-dsvm:ubuntu-hpc:2204:latest"  # VERIFY: has NVIDIA drivers
    disk_gb: int = Field(default=100, ge=30, le=2000)


# Platform id (gpurelay.platforms.catalog) -> its field in BackendsSettings ("lambda" is a
# Python keyword) and backend name.
CLOUD_FIELDS = {
    "runpod": "runpod",
    "vastai": "vastai",
    "lambda": "lambdalabs",
    "digitalocean": "digitalocean",
    "aws": "aws",
    "gcp": "gcp",
    "azure": "azure",
}


class BackendsSettings(_Section):
    """Exactly one section per backend type: one account per service (hard rule 1)."""

    local: LocalSettings = Field(default_factory=LocalSettings)
    kaggle: KaggleSettings = Field(default_factory=KaggleSettings)
    simulated: list[SimulatedSettings] = Field(default_factory=list)
    colab_manual: ColabSettings = Field(default_factory=ColabSettings)
    lightning: LightningSettings = Field(default_factory=LightningSettings)
    modal: ModalSettings = Field(default_factory=ModalSettings)
    runpod: RunPodSettings = Field(default_factory=RunPodSettings)
    vastai: VastSettings = Field(default_factory=VastSettings)
    lambdalabs: LambdaSettings = Field(default_factory=LambdaSettings)
    digitalocean: DigitalOceanSettings = Field(default_factory=DigitalOceanSettings)
    aws: AwsSettings = Field(default_factory=AwsSettings)
    gcp: GcpSettings = Field(default_factory=GcpSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)

    def cloud(self, platform_id: str) -> CloudSettings:
        """The settings of paid platform ``platform_id`` (a catalog id)."""
        value: CloudSettings = getattr(self, CLOUD_FIELDS[platform_id])
        return value

    @model_validator(mode="after")
    def _check_simulated(self) -> BackendsSettings:
        return _unique_simulated(self)


def _unique_simulated(settings: BackendsSettings) -> BackendsSettings:
    names = [sim.name for sim in settings.simulated]
    if len(names) != len(set(names)):
        raise ValueError("simulated backend names must be unique")
    return settings


class NotifySettings(_Section):
    """Notifications (spec §13.6)."""

    desktop: bool = True
    ntfy_topic: str | None = None


class AiFixSettings(_Section):
    """Fix it with AI (spec §20): automatic reruns (A2) and the models tried (A3)."""

    auto: bool = False  # fix and rerun failed jobs without asking (off: the user approves)
    max_attempts: int = Field(default=2, ge=1, le=5)  # AI fixes per original job
    max_gpu_hours: float = Field(default=6.0, gt=0, le=200)  # all attempts of a job together
    claude_fast: str = "sonnet"  # Claude Code model aliases (`claude --model`)
    claude_strong: str = "opus"
    codex_model: str | None = None  # None: Codex's recommended model


class PlatformAnswer(_Section):
    """What the user said about one GPU platform (spec §21.1)."""

    has: Literal["yes", "no", "later"] | None = None  # an account (paid GPU access for clouds)
    updated_at: float | None = None


class PlatformsSettings(_Section):
    """The user's GPU platforms (spec §21.1): their answers, never a credential. Keys of
    ``answers`` are platform ids from :mod:`gpurelay.platforms.catalog`."""

    asked_at: float | None = None  # first answered or skipped: not asked again at sign-in
    answers: dict[str, PlatformAnswer] = Field(default_factory=dict)
    others: list[str] = Field(default_factory=list, max_length=20)  # GPU clouds not listed


class Config(_Section):
    """Top-level configuration."""

    version: int = CONFIG_SCHEMA_VERSION
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    backends: BackendsSettings = Field(default_factory=BackendsSettings)
    notify: NotifySettings = Field(default_factory=NotifySettings)
    aifix: AiFixSettings = Field(default_factory=AiFixSettings)
    platforms: PlatformsSettings = Field(default_factory=PlatformsSettings)


# ====================================================================== paths


def gpurelay_home() -> Path:
    """Directory holding config, state DB, and local data (``$GPURELAY_HOME``)."""
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_HOME_DIRNAME


def default_config_path() -> Path:
    """Path of the config file under :func:`gpurelay_home`."""
    return gpurelay_home() / CONFIG_FILENAME


# ====================================================================== load / save


def load_config(path: Path | None = None) -> Config:
    """Load the config file, or return defaults when it does not exist.

    Raises :class:`ConfigError` on malformed TOML or invalid values.
    """
    path = path or default_config_path()
    if not path.exists():
        return Config()
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid configuration:\n{exc}") from exc


def save_config(config: Config, path: Path | None = None) -> Path:
    """Atomically write ``config`` as TOML and return the path written."""
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_none=True)
    text = tomli_w.dumps(data)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path
