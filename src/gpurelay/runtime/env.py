"""Environment fingerprint recorded with every checkpoint and segment (spec §12.1)."""

from __future__ import annotations

import platform
import sys
from typing import Any

import gpurelay

_VERSIONED_LIBS = ("numpy", "torch", "tensorflow", "transformers", "lightning", "keras")


def env_fingerprint() -> dict[str, Any]:
    """Python/platform/library versions and GPU models. Only inspects loaded libraries."""
    fingerprint: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpurelay": gpurelay.__version__,
        "libraries": {},
        "gpus": [],
    }
    for name in _VERSIONED_LIBS:
        module = sys.modules.get(name)
        if module is not None:
            fingerprint["libraries"][name] = str(getattr(module, "__version__", "unknown"))

    torch = sys.modules.get("torch")
    if torch is not None:
        fingerprint["cuda"] = getattr(torch.version, "cuda", None)
        try:
            if torch.cuda.is_available():
                fingerprint["gpus"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
        except Exception:  # pragma: no cover - driver issues
            pass
    return fingerprint
