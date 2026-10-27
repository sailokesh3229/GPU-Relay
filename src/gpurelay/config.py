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


_RESERVED_BACKEND_NAMES = frozenset({"local", "kaggle", "colab_manual", "lightning", "modal"})


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


class BackendsSettings(_Section):
    """Exactly one section per backend type: one account per service (hard rule 1)."""

    local: LocalSettings = Field(default_factory=LocalSettings)
    kaggle: KaggleSettings = Field(default_factory=KaggleSettings)
    simulated: list[SimulatedSettings] = Field(default_factory=list)
    colab_manual: ColabSettings = Field(default_factory=ColabSettings)
    lightning: LightningSettings = Field(default_factory=LightningSettings)
    modal: ModalSettings = Field(default_factory=ModalSettings)

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


class Config(_Section):
    """Top-level configuration."""

    version: int = CONFIG_SCHEMA_VERSION
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    backends: BackendsSettings = Field(default_factory=BackendsSettings)
    notify: NotifySettings = Field(default_factory=NotifySettings)


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
