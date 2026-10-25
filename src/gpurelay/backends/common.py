"""Helpers shared by backends whose sessions use Hugging Face Hub storage (Kaggle, Colab)."""

from __future__ import annotations

from gpurelay.config import RuntimeSettings

# The Hub allows 128 commits per hour per repo. In a Hub-backed session every heartbeat is a
# commit and every checkpoint two (data, then LATEST), so both get floors here:
# 12 heartbeats + at most 24 checkpoint commits an hour per session, which leaves room for the
# controller (lease, state backup) and a second session.
MIN_HEARTBEAT_S = 300.0
MIN_CHECKPOINT_INTERVAL_S = 300.0
STALE_HEARTBEATS = 3  # a session is stale after this many missed heartbeats


def session_settings(runtime: RuntimeSettings) -> RuntimeSettings:
    """Runtime settings for Hub-backed sessions (heartbeats and checkpoints are commits).

    The checkpoint floor never exceeds ``max_lost_work_s``: the lost-work limit still wins."""
    heartbeat = max(runtime.heartbeat_s, MIN_HEARTBEAT_S)
    floor = min(MIN_CHECKPOINT_INTERVAL_S, runtime.max_lost_work_s)
    return runtime.model_copy(
        update={
            "heartbeat_s": heartbeat,
            "stale_heartbeat_s": max(runtime.stale_heartbeat_s, STALE_HEARTBEATS * heartbeat),
            "min_interval_s": max(runtime.min_interval_s, floor),
        }
    )
