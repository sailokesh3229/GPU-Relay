"""Mid-epoch DataLoader fast-forward (Phase 9 item 2): skip batches without loading them."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from gpurelay.runtime.session import _sampler_driven, _skip_batches  # noqa: E402

pytestmark = pytest.mark.torch


class Counting(Dataset):  # type: ignore[type-arg]
    def __init__(self, n: int) -> None:
        self.n = n
        self.loaded = 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Any:
        self.loaded += 1
        return torch.tensor([i])


def _batches_after_skip(skip: int) -> tuple[list[int], int]:
    data = Counting(64)
    torch.manual_seed(3)
    loader = DataLoader(data, batch_size=8, shuffle=True)
    it = iter(loader)
    assert _skip_batches(it, skip)
    loaded_while_skipping = data.loaded
    return next(it).flatten().tolist(), loaded_while_skipping


def test_skipping_loads_nothing_and_lands_on_the_same_batch() -> None:
    data = Counting(64)
    torch.manual_seed(3)
    expected = [b.flatten().tolist() for b in DataLoader(data, batch_size=8, shuffle=True)]
    batch, loaded = _batches_after_skip(5)
    assert loaded == 0 and batch == expected[5]


def test_only_single_process_map_style_loaders_are_skipped() -> None:
    data = Counting(16)
    assert _sampler_driven(DataLoader(data, batch_size=4)) and not _sampler_driven(range(3))
    assert not _skip_batches(iter([1, 2, 3]), 1)
    assert not _skip_batches(iter(DataLoader(data, batch_size=4, num_workers=1)), 1)
