"""Helpers shared by backends whose sessions use Hugging Face Hub storage (Kaggle, Colab)."""

from __future__ import annotations

from gpurelay.config import RuntimeSettings

MIN_HEARTBEAT_S = 120.0  # every heartbeat is a Hub commit


def session_settings(runtime: RuntimeSettings) -> RuntimeSettings:
    """Runtime settings for Hub-backed sessions (heartbeats are commits there)."""
    return runtime.model_copy(update={"heartbeat_s": max(runtime.heartbeat_s, MIN_HEARTBEAT_S)})
