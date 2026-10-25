"""Budgets across platforms (plan P11): units, GPU speed, daily windows and the order of use.

"Done when" (roadmap): simulated tests choose the platform that finishes a job soonest
without wasting expiring credits.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpurelay.config import Config, SimulatedSettings, StorageSettings
from gpurelay.controller.db import StateDB
from gpurelay.controller.gpus import normalize, speed_of
from gpurelay.controller.quota import Budget, QuotaLedger
from gpurelay.controller.registry import build_backends
from gpurelay.controller.scheduler import BackendView, JobView, choose, plan
from gpurelay.models import BudgetModel
from gpurelay.storage.localfs import LocalFSStorage

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
MONTH_END = datetime(2026, 11, 1, tzinfo=timezone.utc)


def view(name: str, remaining: float | None = 10.0, *, model: str = "weekly_rolling",
         speed: float = 1.0, unlimited: bool = False, reset: datetime | None = None,
         gpu: str | None = None) -> BackendView:  # fmt: skip
    return BackendView(
        name=name, automated=True, ready=True, remaining_h=remaining, unlimited=unlimited,
        max_concurrent=1, running=0, max_session_s=12 * 3600, gpu_label=gpu,
        next_reset=reset, budget_model=model, speed=speed,
    )  # fmt: skip


JOB = JobView("job-1", 0, NOW)


@pytest.mark.parametrize(
    ("label", "name", "speed"),
    [
        ("NvidiaTeslaT4", "t4", 1.0),
        ("Tesla T4", "t4", 1.0),
        ("NvidiaTeslaP100", "p100", 1.3),
        ("L4", "l4", 1.8),
        ("A100-40GB", "a100", 5.0),
        ("nvidia a10g", "a10g", 2.2),
        ("cpu", "cpu", 0.05),
        ("SimGPU", None, 1.0),
        (None, None, 1.0),
    ],
)
def test_gpu_names_and_speeds(label: str | None, name: str | None, speed: float) -> None:
    assert normalize(label) == name
    assert speed_of(label) == speed


def test_budgets_in_dollars_and_credits_convert_to_gpu_hours() -> None:
    modal = Budget.from_amount(BudgetModel.MONTHLY, 30.0, "usd", 0.59)
    assert modal.hours == pytest.approx(50.85, abs=0.01)
    assert modal.amount(10.0) == pytest.approx(5.90)
    lightning = Budget.from_amount(BudgetModel.MONTHLY, 15.0, "credits", 0.19)
    assert lightning.unit == "credits" and lightning.hours == pytest.approx(78.9, abs=0.1)
    with pytest.raises(ValueError):
        Budget.from_amount(BudgetModel.MONTHLY, 30.0, "euro", 0.5)
    with pytest.raises(ValueError):
        Budget.from_amount(BudgetModel.MONTHLY, 30.0, "usd", 0)


def test_daily_budget_counts_only_today_and_resets_at_midnight(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    now = NOW.timestamp()
    db.add_usage("x", now - 26 * 3600, now - 25 * 3600, "seg-old", now)  # yesterday
    db.add_usage("x", now - 3 * 3600, now - 2 * 3600, "seg-today", now)  # today, 1 h
    state = QuotaLedger(db).state("x", Budget(BudgetModel.DAILY, 4.0, estimate=False), now)
    assert state.used_h == pytest.approx(1.0) and state.remaining_h == pytest.approx(3.0)
    assert state.next_reset == datetime(2026, 10, 26, tzinfo=timezone.utc)


def test_expiring_credits_go_first_on_a_comparable_gpu() -> None:
    weekly = view("kaggle", 20.0, gpu="T4")
    monthly = view("modal", 3.0, model="monthly", reset=MONTH_END, gpu="T4")
    assert choose([weekly, monthly], JOB).name == "modal"  # less left, but lost at reset


def test_a_much_slower_expiring_budget_does_not_win() -> None:
    weekly = view("kaggle", 20.0, speed=1.0)
    slow = view("credits-cpu", 50.0, model="monthly", reset=MONTH_END, speed=0.3)
    assert choose([weekly, slow], JOB).name == "kaggle"  # 0.3 < half of 1.0: too slow


def test_the_fastest_gpu_finishes_soonest() -> None:
    t4 = view("kaggle", 25.0, speed=1.0)
    l4 = view("other", 6.0, speed=1.8)
    assert choose([t4, l4], JOB).name == "other"
    cpu = view("local", None, unlimited=True, speed=0.05)
    assert choose([cpu, t4], JOB).name == "kaggle"  # unlimited CPU never beats a GPU


def test_of_two_expiring_budgets_the_one_resetting_first_goes_first() -> None:
    soon = view("daily", 2.0, model="daily", reset=NOW + timedelta(hours=12))
    later = view("monthly", 9.0, model="monthly", reset=MONTH_END)
    assert choose([later, soon], JOB).name == "daily"


def test_equal_platforms_keep_the_old_rules() -> None:
    a, b = view("a", 10.0, gpu="T4"), view("b", 12.0, gpu="P100x")
    assert choose([a, b], JOB).name == "b"  # most remaining
    near = view("b", 10.3, gpu="P100x")
    resumed = JobView("job-1", 0, NOW, previous_gpus=["T4"])
    assert choose([a, near], resumed).name == "a"  # tie: same GPU as before
    preferred = JobView("job-1", 0, NOW, prefer_backend="a")
    assert choose([a, b], preferred).name == "a"


def test_plan_spreads_jobs_without_wasting_credits() -> None:
    """Two jobs, one slot each: the expiring T4 credits and the fast weekly L4 both work."""
    jobs = [JobView("j1", 1, NOW), JobView("j2", 0, NOW)]
    backends = [
        view("modal", 5.0, model="monthly", reset=MONTH_END, speed=1.0),
        view("fast", 30.0, speed=1.8),
    ]
    decisions = plan(jobs, backends, min_segment_h=0.5)
    assert [(d.job_id, d.action, d.backend) for d in decisions] == [
        ("j1", "launch", "modal"),  # 1.0 >= 0.5 * 1.8: credits first
        ("j2", "launch", "fast"),
    ]
    empty = view("modal", 0.2, model="monthly", reset=MONTH_END)  # below a minimum segment
    [only] = plan([JobView("j3", 0, NOW)], [empty, view("fast", 30.0, speed=1.8)],
                  min_segment_h=0.5)  # fmt: skip
    assert only.backend == "fast"


def test_registry_builds_unit_budgets_and_speeds(tmp_path: Path) -> None:
    cfg = Config(storage=StorageSettings(kind="localfs", localfs_root=tmp_path / "store"))
    cfg.backends.local.enabled = True
    cfg.backends.kaggle.enabled = False
    cfg.backends.simulated = [
        SimulatedSettings(name="modal-sim", gpu_label="L4", budget_hours=30.0,
                          budget_model=BudgetModel.MONTHLY, budget_unit="usd",
                          cost_per_hour=0.80),
    ]  # fmt: skip
    slots = build_backends(cfg, LocalFSStorage(tmp_path / "store"), tmp_path / "sessions")
    sim = slots["modal-sim"]
    assert sim.budget.unit == "usd" and sim.budget.hours == pytest.approx(37.5)
    assert sim.speed == 1.8 and slots["local"].speed == 0.05
