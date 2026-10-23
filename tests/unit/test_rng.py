from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from gpurelay.runtime.rng import capture_rng, load_rng, restore_rng, save_rng


def _draws() -> list[object]:
    out: list[object] = [random.random(), random.randint(0, 10**9)]
    np = sys.modules.get("numpy")
    if np is not None:
        out.append(np.random.rand(4).tolist())
    torch = sys.modules.get("torch")
    if torch is not None:
        out.append(torch.rand(4).tolist())
    return out


def test_python_rng_round_trip() -> None:
    random.seed(7)
    state = capture_rng()
    first = [random.random() for _ in range(5)]
    random.seed(999)
    restore_rng(state)
    assert [random.random() for _ in range(5)] == first


def test_round_trip_through_file(tmp_path: Path) -> None:
    random.seed(3)
    save_rng(tmp_path / "rng.pt")
    first = _draws()
    random.seed(4)
    assert load_rng(tmp_path / "rng.pt") == []
    assert _draws() == first


def test_unimported_libraries_are_not_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("numpy", "torch", "tensorflow"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    state = capture_rng()
    assert set(state) == {"format", "python"}


def test_numpy_rng_round_trip() -> None:
    np = pytest.importorskip("numpy")
    np.random.seed(11)
    state = capture_rng()
    first = np.random.rand(8).tolist()
    np.random.seed(0)
    restore_rng(state)
    assert np.random.rand(8).tolist() == first


@pytest.mark.torch
def test_torch_cpu_rng_round_trip() -> None:
    import torch

    torch.manual_seed(5)
    state = capture_rng()
    first = torch.randn(16)
    dropout_mask = torch.nn.functional.dropout(torch.ones(64), p=0.5)
    torch.manual_seed(6)
    restore_rng(state)
    assert torch.equal(torch.randn(16), first)
    assert torch.equal(torch.nn.functional.dropout(torch.ones(64), p=0.5), dropout_mask)
