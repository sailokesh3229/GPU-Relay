"""Shared pytest fixtures and marker handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gpurelay.constants import ENV_HOME

# Marker name -> module that must be importable for the test to run.
_OPTIONAL_LIBS = {
    "torch": "torch",
    "tf": "tensorflow",
    "hf": "transformers",
    "lightning": "lightning",
    "keras": "keras",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked for an optional ML library that is not installed."""
    for marker, module in _OPTIONAL_LIBS.items():
        if importlib.util.find_spec(module) is not None:
            continue
        skip = pytest.mark.skip(reason=f"{module} not installed")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def relay_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$GPURELAY_HOME`` at a per-test temp dir so tests never touch ``~``."""
    home = tmp_path / "gpurelay-home"
    monkeypatch.setenv(ENV_HOME, str(home))
    return home
