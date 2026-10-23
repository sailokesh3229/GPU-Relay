"""Framework hooks degrade (warn, do not patch) on library versions they do not support."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from gpurelay.runtime.hooks import hf, keras, lightning, major_version


class _FakeSession:
    runtime_dir = Path(".")


def _fake_module(name: str, version: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__version__ = version  # type: ignore[attr-defined]
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_major_version() -> None:
    assert major_version("4.57.1") == 4
    assert major_version("dev") is None


def test_keras_2_is_not_patched(caplog: pytest.LogCaptureFixture) -> None:
    model_cls = type("Model", (), {"fit": lambda self: "original"})
    fake = _fake_module("keras", "2.15.0", Model=model_cls)
    keras.patch(fake, _FakeSession())  # type: ignore[arg-type]
    assert model_cls().fit() == "original"
    assert "not supported" in caplog.text


def test_lightning_1_is_not_patched(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pytorch_lightning", _fake_module("pytorch_lightning", "1.9"))
    trainer_cls = type("Trainer", (), {"fit": lambda self: "original"})
    module = _fake_module("pytorch_lightning.trainer.trainer", "1.9", Trainer=trainer_cls)
    lightning.patch(module, _FakeSession())  # type: ignore[arg-type]
    assert trainer_cls().fit() == "original"
    assert "not supported" in caplog.text


def test_transformers_3_is_not_patched(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_module("transformers", "3.5.1"))
    trainer_cls = type("Trainer", (), {"train": lambda self: "original"})
    module = _fake_module("transformers.trainer", "3.5.1", Trainer=trainer_cls)
    hf.patch(module, _FakeSession())  # type: ignore[arg-type]
    assert trainer_cls().train() == "original"
    assert "not supported" in caplog.text
