"""Capture and restore global random-number-generator state (spec §7.5).

Covers Python ``random``, the NumPy global RNG, torch CPU and every CUDA device, and the
TensorFlow global generator. Optional libraries are handled lazily:

* on capture, a library is included only if user code has already imported it
  (if it was never imported, its RNG cannot have influenced the run);
* on restore, a library present in the saved state is imported so that its module-level
  RNG is set before user code uses it.
"""

from __future__ import annotations

import importlib
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RNG_FORMAT_VERSION = 1


def capture_rng() -> dict[str, Any]:
    """Return a picklable dict of all available global RNG states."""
    state: dict[str, Any] = {"format": RNG_FORMAT_VERSION, "python": random.getstate()}

    np = sys.modules.get("numpy")
    if np is not None:
        state["numpy"] = np.random.get_state()

    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()

    tf = sys.modules.get("tensorflow")
    if tf is not None:
        try:
            state["tf_global_generator"] = tf.random.get_global_generator().state.numpy()
        except Exception as exc:  # best effort: TF APIs vary by version
            log.warning("could not capture TensorFlow RNG state: %s", exc)

    return state


def restore_rng(state: dict[str, Any]) -> list[str]:
    """Restore RNG states captured by :func:`capture_rng`.

    Returns warnings for states that could not be restored (e.g. CUDA states saved on a
    machine with a different device count).
    """
    warnings: list[str] = []
    random.setstate(state["python"])

    if "numpy" in state:
        np = importlib.import_module("numpy")
        np.random.set_state(state["numpy"])

    if "torch_cpu" in state or "torch_cuda" in state:
        torch = importlib.import_module("torch")
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state:
            saved = state["torch_cuda"]
            if not torch.cuda.is_available():
                warnings.append("CUDA RNG state saved but CUDA is not available; skipped")
            else:
                count = torch.cuda.device_count()
                if count != len(saved):
                    warnings.append(
                        f"CUDA RNG saved for {len(saved)} device(s), restoring onto {count}"
                    )
                for device, device_state in enumerate(saved[:count]):
                    torch.cuda.set_rng_state(device_state, device)

    if "tf_global_generator" in state:
        try:
            tf = importlib.import_module("tensorflow")
            tf.random.get_global_generator().reset(state["tf_global_generator"])
        except Exception as exc:
            warnings.append(f"could not restore TensorFlow RNG state: {exc}")

    for message in warnings:
        log.warning(message)
    return warnings


def save_rng(path: Path) -> None:
    """Capture RNG state and write it to ``path`` (pickle format)."""
    with open(path, "wb") as fh:
        pickle.dump(capture_rng(), fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_rng(path: Path) -> list[str]:
    """Read RNG state from ``path`` and restore it. Returns warnings."""
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    return restore_rng(state)


def seed_all(seed: int) -> None:
    """Seed every RNG that is in use (Python, NumPy, torch incl. CUDA, TensorFlow).

    Used for ``relay.chunks(..., parallel=True)``: each chunk starts from a seed derived
    from its id, so its result does not depend on where or after what it runs.
    """
    seed %= 2**32
    random.seed(seed)
    np = sys.modules.get("numpy")
    if np is not None:
        np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)  # also seeds every CUDA device
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        try:
            tf.random.set_seed(seed)
        except Exception as exc:  # pragma: no cover - TF optional
            log.debug("tensorflow seed failed: %s", exc)


def chunk_seed(base: int, chunk_id: int) -> int:
    """Seed of chunk ``chunk_id`` in a parallel chunks loop with seed ``base``."""
    return (base * 1_000_003 + chunk_id) % 2**32
