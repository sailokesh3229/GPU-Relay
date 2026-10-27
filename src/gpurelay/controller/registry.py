"""Backends configured for the controller, with their budgets (spec §10, §13.2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurelay.backends.local import LocalBackend
from gpurelay.backends.simulated import SimulatedBackend
from gpurelay.config import Config
from gpurelay.controller.gpus import speed_of
from gpurelay.controller.quota import Budget
from gpurelay.models import BudgetModel
from gpurelay.storage.base import Storage

log = logging.getLogger(__name__)


@dataclass
class BackendSlot:
    """A backend plus what the scheduler needs to know about it."""

    name: str
    backend: Any  # Backend protocol (+ ``returncode`` for subprocess backends)
    budget: Budget
    automated: bool
    max_concurrent: int
    max_session_s: float
    gpu_label: str | None
    speed: float = 1.0  # relative GPU speed, T4 = 1.0 (controller/gpus.py)
    # Paid GPU time after the free budget: USD per GPU hour (None: no paid option) and the
    # owner's monthly limit for it (0: never pay).
    paid_usd_per_h: float | None = None
    paid_cap_usd: float = 0.0
    # Manual backends: asked before the automated ones, and how long a request waits for
    # the user before the job goes elsewhere (None: it waits until the user acts).
    first: bool = False
    user_wait_s: float | None = None


def build_backends(config: Config, storage: Storage, sessions_root: Path) -> dict[str, BackendSlot]:
    """One slot per configured backend. Hard rule 1: one account per service type."""
    slots: dict[str, BackendSlot] = {}
    local = config.backends.local
    if local.enabled:
        from gpurelay.backends.local_env import frozen, runtime_python, system_python

        home = sessions_root.parent
        gpu = local.gpu_label or (detect_local_gpu() if local.detect_gpu else None)
        packaged = frozen()
        slots["local"] = BackendSlot(
            name="local",
            backend=LocalBackend(
                storage,
                sessions_root / "local",
                name="local",
                settings=config.runtime,
                max_session_s=local.max_session_s,
                gpu_label=gpu,
                python_resolver=(lambda: runtime_python(home)) if packaged else None,
                install_requirements=packaged,
                ready_check=(
                    (
                        lambda: (
                            None
                            if system_python() or (home / "local-runtime").exists()
                            else "install Python 3.10+ (python.org) to run jobs on this computer"
                        )
                    )
                    if packaged
                    else None
                ),
            ),
            budget=Budget(BudgetModel.UNKNOWN_OPPORTUNISTIC, None, unlimited=True),
            automated=True,
            max_concurrent=local.max_concurrent,
            max_session_s=local.max_session_s,
            gpu_label=gpu,
            speed=speed_of(gpu or "cpu"),  # no GPU found: this computer's CPU
        )
    for sim in config.backends.simulated:
        slots[sim.name] = BackendSlot(
            name=sim.name,
            backend=SimulatedBackend(
                storage,
                sessions_root / sim.name,
                name=sim.name,
                settings=config.runtime,
                max_session_s=sim.max_session_s,
                gpu_label=sim.gpu_label,
                kill_after_s=sim.kill_after_s,
                kill_between_s=sim.kill_between_s,
                seed=sim.seed,
            ),
            budget=(
                Budget(sim.budget_model, None, unlimited=True, estimate=False)
                if sim.budget_hours is None
                else Budget.from_amount(
                    sim.budget_model,
                    sim.budget_hours,
                    sim.budget_unit,
                    sim.cost_per_hour,
                    estimate=False,
                )
            ),
            automated=True,
            max_concurrent=sim.max_concurrent,
            max_session_s=sim.max_session_s,
            gpu_label=sim.gpu_label,
            speed=speed_of(sim.gpu_label),
            paid_usd_per_h=sim.paid_usd_per_h,
            paid_cap_usd=sim.monthly_paid_cap_usd,
        )
    kaggle = config.backends.kaggle
    if kaggle.enabled:
        from gpurelay.backends.kaggle import KaggleBackend

        slots["kaggle"] = BackendSlot(
            name="kaggle",
            backend=KaggleBackend(
                settings=kaggle,
                runtime=config.runtime,
                storage_uri=_storage_uri(storage),
                work_root=sessions_root / "kaggle",
                wheel_for=_wheel_reader(storage),
            ),
            budget=Budget(kaggle.budget_model, kaggle.weekly_gpu_hours, estimate=True),
            automated=True,
            max_concurrent=kaggle.max_concurrent,
            max_session_s=kaggle.max_session_s,
            gpu_label=kaggle.accelerator if kaggle.enable_gpu else "cpu",
            speed=speed_of(kaggle.accelerator if kaggle.enable_gpu else "cpu"),
        )
    lightning = config.backends.lightning
    if lightning.enabled:
        from gpurelay.backends.lightning import LightningBackend

        slots["lightning"] = BackendSlot(
            name="lightning",
            backend=LightningBackend(
                settings=lightning,
                runtime=config.runtime,
                storage_uri=_storage_uri(storage),
                wheel_for=_wheel_reader(storage),
            ),
            budget=(
                Budget.from_amount(
                    lightning.budget_model,
                    lightning.monthly_credits,
                    "credits",
                    lightning.credits_per_hour,
                    estimate=True,
                )
                if lightning.monthly_credits is not None and lightning.credits_per_hour
                else Budget(lightning.budget_model, lightning.monthly_gpu_hours, estimate=True)
            ),
            automated=True,
            max_concurrent=lightning.max_concurrent,
            max_session_s=lightning.max_session_s,
            gpu_label=lightning.machine,
            speed=speed_of(lightning.machine),
        )
    modal = config.backends.modal
    if modal.enabled:
        from gpurelay.backends.modal import ModalBackend

        slots["modal"] = BackendSlot(
            name="modal",
            backend=ModalBackend(
                settings=modal,
                runtime=config.runtime,
                storage_uri=_storage_uri(storage),
                wheel_for=_wheel_reader(storage),
            ),
            budget=Budget.from_amount(
                modal.budget_model,
                modal.monthly_credit_usd,
                "usd",
                modal.usd_per_hour,
                estimate=True,
            ),
            automated=True,
            max_concurrent=modal.max_concurrent,
            max_session_s=modal.max_session_s,
            gpu_label=modal.gpu,
            speed=speed_of(modal.gpu),
            paid_usd_per_h=modal.usd_per_hour,
            paid_cap_usd=modal.monthly_paid_cap_usd,
        )
    colab = config.backends.colab_manual
    if colab.enabled:
        from gpurelay.backends.colab_manual import ColabManualBackend

        slots["colab_manual"] = BackendSlot(
            name="colab_manual",
            backend=ColabManualBackend(
                settings=colab,
                runtime=config.runtime,
                storage=storage,
                storage_uri=_storage_uri(storage),
                work_root=sessions_root / "colab_manual",
                wheel_for=_wheel_reader(storage),
            ),
            budget=Budget(colab.budget_model, None, estimate=True),
            automated=False,  # hard rule 2
            max_concurrent=1,
            max_session_s=colab.max_session_s,
            gpu_label=colab.gpu_label,
            speed=speed_of(colab.gpu_label or "T4"),  # free Colab is usually a T4
            first=colab.first,
            user_wait_s=float(colab.wait_s) if colab.first else None,
        )
    return slots


def detect_local_gpu() -> str | None:
    """This computer's NVIDIA GPU (plan P15); a separate name so tests can replace it."""
    from gpurelay.backends.local_env import detect_gpu

    return detect_gpu()


def _storage_uri(storage: Storage) -> str:
    from gpurelay.storage.factory import storage_uri

    try:
        return storage_uri(storage)
    except ValueError:
        return ""


def _wheel_reader(storage: Storage) -> Any:
    from gpurelay.bundle.packager import read_manifest

    def wheel_for(job_id: str) -> str | None:
        return read_manifest(storage, job_id).runtime_wheel

    return wheel_for
