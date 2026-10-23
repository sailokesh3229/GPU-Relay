"""Backends configured for the controller, with their budgets (spec §10, §13.2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurelay.backends.local import LocalBackend
from gpurelay.backends.simulated import SimulatedBackend
from gpurelay.config import Config
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


def build_backends(config: Config, storage: Storage, sessions_root: Path) -> dict[str, BackendSlot]:
    """One slot per configured backend. Hard rule 1: one account per service type."""
    slots: dict[str, BackendSlot] = {}
    local = config.backends.local
    if local.enabled:
        slots["local"] = BackendSlot(
            name="local",
            backend=LocalBackend(
                storage,
                sessions_root / "local",
                name="local",
                settings=config.runtime,
                max_session_s=local.max_session_s,
                gpu_label=local.gpu_label,
            ),
            budget=Budget(BudgetModel.UNKNOWN_OPPORTUNISTIC, None, unlimited=True),
            automated=True,
            max_concurrent=local.max_concurrent,
            max_session_s=local.max_session_s,
            gpu_label=local.gpu_label,
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
            budget=Budget(
                sim.budget_model,
                sim.budget_hours,
                unlimited=sim.budget_hours is None,
                estimate=False,
            ),
            automated=True,
            max_concurrent=sim.max_concurrent,
            max_session_s=sim.max_session_s,
            gpu_label=sim.gpu_label,
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
            budget=Budget(lightning.budget_model, lightning.monthly_gpu_hours, estimate=True),
            automated=True,
            max_concurrent=lightning.max_concurrent,
            max_session_s=lightning.max_session_s,
            gpu_label=lightning.machine,
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
        )
    return slots


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
