"""Kill-and-resume equivalence for torch state (plan Phase 1 verification).

An uninterrupted training run is compared with a run that checkpoints mid-way and
resumes in a *fresh* namespace whose objects were rebuilt with a different seed (as
replay would rebuild them) — results must be bit-identical on CPU.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from gpurelay.retry import RetryPolicy  # noqa: E402
from gpurelay.runtime.checkpoint import CheckpointManager, load_latest_checkpoint  # noqa: E402
from gpurelay.storage import LocalFSStorage  # noqa: E402

pytestmark = pytest.mark.torch

JOB = "job-torch"
TOTAL_STEPS = 12
CUT = 5
FAST_RETRY = RetryPolicy(attempts=2, base_delay_s=0.0, jitter=0.0, sleep=lambda _s: None)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build(init_seed: int) -> dict[str, Any]:
    """What 'replay' produces: fresh objects (deliberately different initial weights)."""
    torch.manual_seed(init_seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8, growth_interval=2)
    return {"model": model, "opt": opt, "sched": sched, "scaler": scaler, "losses": []}


def _train_step(ns: dict[str, Any]) -> None:
    """Uses torch (randn + dropout), NumPy, and Python RNGs."""
    model, opt, sched, scaler = ns["model"], ns["opt"], ns["sched"], ns["scaler"]
    x = torch.randn(32, 8)
    noise = torch.from_numpy(np.random.randn(32, 1)).float()
    y = x.sum(dim=1, keepdim=True) + 0.1 * noise + random.random()
    model.train()
    loss = torch.nn.functional.mse_loss(model(x), y)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()
    ns["losses"].append(loss.item())


def _uninterrupted() -> dict[str, Any]:
    ns = _build(init_seed=0)
    _seed_all(42)
    for _ in range(TOTAL_STEPS):
        _train_step(ns)
    return ns


def _checkpoint_after_cut(tmp_path: Path) -> LocalFSStorage:
    store = LocalFSStorage(tmp_path / "store")
    ns = _build(init_seed=0)
    _seed_all(42)
    for _ in range(CUT):
        _train_step(ns)
    mgr = CheckpointManager(store, job_id=JOB, segment_id="seg-1", local_root=tmp_path / "l1")
    mgr.snapshot(ns, safe_point_id=f"step:{CUT}")
    result = mgr.close()
    assert result is not None and result.ok, result
    # Work done after the checkpoint is lost when the session dies.
    for _ in range(2):
        _train_step(ns)
    return store


def _assert_same_state(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert a["losses"] == b["losses"]
    sd_a, sd_b = a["model"].state_dict(), b["model"].state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), key
    opt_a, opt_b = a["opt"].state_dict()["state"], b["opt"].state_dict()["state"]
    for idx in opt_a:
        for key in opt_a[idx]:
            assert torch.equal(torch.as_tensor(opt_a[idx][key]), torch.as_tensor(opt_b[idx][key]))
    assert a["sched"].get_last_lr() == b["sched"].get_last_lr()
    assert a["scaler"].get_scale() == b["scaler"].get_scale()


def test_resume_into_live_objects_is_identical(tmp_path: Path) -> None:
    expected = _uninterrupted()
    store = _checkpoint_after_cut(tmp_path)

    # "New session": replay rebuilds objects (different init, scrambled RNG) ...
    _seed_all(999)
    ns = _build(init_seed=123)
    live_model, live_opt = ns["model"], ns["opt"]
    loaded = load_latest_checkpoint(store, JOB, tmp_path / "restore", retry=FAST_RETRY)
    assert loaded is not None and loaded.seq == 1
    report = loaded.restore(ns)

    # ... restore loads into the live objects (identity kept) ...
    assert sorted(report.loaded_into_live) == ["model", "opt", "scaler", "sched"]
    assert report.unpickled_torch == []
    assert ns["model"] is live_model and ns["opt"] is live_opt
    assert len(ns["losses"]) == CUT

    for _ in range(TOTAL_STEPS - CUT):
        _train_step(ns)
    _assert_same_state(ns, expected)


def test_optimizer_still_drives_live_model_after_restore(tmp_path: Path) -> None:
    store = _checkpoint_after_cut(tmp_path)
    ns = _build(init_seed=7)
    loaded = load_latest_checkpoint(store, JOB, tmp_path / "restore", retry=FAST_RETRY)
    assert loaded is not None
    loaded.restore(ns)

    params = list(ns["model"].parameters())
    assert all(a is b for a, b in zip(ns["opt"].param_groups[0]["params"], params, strict=True))
    before = [p.detach().clone() for p in params]
    _train_step(ns)
    assert any(not torch.equal(b, p) for b, p in zip(before, params, strict=True))


def test_whole_object_fallback_without_live_objects(tmp_path: Path) -> None:
    expected = _uninterrupted()
    store = _checkpoint_after_cut(tmp_path)

    ns: dict[str, Any] = {}  # nothing was replayed: no live objects exist
    loaded = load_latest_checkpoint(store, JOB, tmp_path / "restore", retry=FAST_RETRY)
    assert loaded is not None
    report = loaded.restore(ns)
    assert sorted(report.unpickled_torch) == ["model", "opt", "scaler", "sched"]

    # Unpickled together, so the optimizer references the unpickled model's parameters.
    params = list(ns["model"].parameters())
    assert all(a is b for a, b in zip(ns["opt"].param_groups[0]["params"], params, strict=True))
    assert ns["sched"].optimizer is ns["opt"]

    for _ in range(TOTAL_STEPS - CUT):
        _train_step(ns)
    _assert_same_state(ns, expected)


def test_repeated_interruptions_are_identical(tmp_path: Path) -> None:
    """Checkpoint after every 3 steps; each resume happens in a brand-new namespace."""
    expected = _uninterrupted()
    store = LocalFSStorage(tmp_path / "store")
    ns = _build(init_seed=0)
    _seed_all(42)
    segment = 0
    while len(ns["losses"]) < TOTAL_STEPS:
        for _ in range(min(3, TOTAL_STEPS - len(ns["losses"]))):
            _train_step(ns)
        segment += 1
        mgr = CheckpointManager(
            store, job_id=JOB, segment_id=f"seg-{segment}", local_root=tmp_path / "local"
        )
        mgr.snapshot(ns, safe_point_id=f"step:{len(ns['losses'])}")
        result = mgr.close()
        assert result is not None and result.ok
        # Session dies; the next one rebuilds and restores.
        _seed_all(segment * 1000)
        ns = _build(init_seed=segment)
        loaded = load_latest_checkpoint(store, JOB, tmp_path / f"r{segment}", retry=FAST_RETRY)
        assert loaded is not None
        loaded.restore(ns)
    _assert_same_state(ns, expected)


def test_layout_contains_torch_state(tmp_path: Path) -> None:
    store = _checkpoint_after_cut(tmp_path)
    names = [k.rsplit("/", 1)[-1] for k in store.list(f"jobs/{JOB}/checkpoints/ckpt-1/")]
    assert sorted(names) == [
        "manifest.json",
        "meta.json",
        "rng.pt.zst",
        "torch_state.pt.zst",
        "vars.pkl.zst",
    ]
