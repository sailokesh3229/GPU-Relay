"""GPU Relay: run ML workloads to completion on free GPU tiers.

The manual SDK is re-exported here::

    import gpurelay as relay
    for epoch in relay.range(100): ...
"""

__version__ = "0.1.0.dev0"

from gpurelay.sdk import (
    Chunk,
    State,
    chunks,
    is_active,
    load_results,
    range,
    safe_point,
    save_result,
    state,
)

__all__ = [
    "Chunk",
    "State",
    "__version__",
    "chunks",
    "is_active",
    "load_results",
    "range",
    "safe_point",
    "save_result",
    "state",
]
