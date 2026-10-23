"""Layer B: framework hooks (spec §7.2), installed lazily when the framework is imported.

Each hook turns the framework's training call into a resumable event: the framework's
own checkpoint becomes part of the relay checkpoint (``framework/``), saves are requested
when the relay's checkpoint policy says so, and a resumed segment passes the restored
checkpoint back to the framework.

Hooks check the installed library's major version and stay inactive (with a warning) on
versions they were not written for.
"""

from __future__ import annotations

import functools
import logging
import types
from collections.abc import Callable
from typing import TYPE_CHECKING

from gpurelay.runtime.importhooks import ImportHook, when_imported

if TYPE_CHECKING:
    from gpurelay.runtime.session import RelaySession

log = logging.getLogger(__name__)


def major_version(version: str) -> int | None:
    try:
        return int(version.split(".")[0])
    except ValueError:
        return None


def _apply(
    patch: Callable[[types.ModuleType, RelaySession], None],
    session: RelaySession,
    module: types.ModuleType,
) -> None:
    patch(module, session)


def install_hooks(session: RelaySession) -> list[ImportHook]:
    """Register all framework hooks; returns the pending import hooks (for removal)."""
    from gpurelay.runtime.hooks import hf, keras, lightning

    pending: list[ImportHook] = []
    for module_name, patch in (
        ("transformers.trainer", hf.patch),
        ("lightning.pytorch.trainer.trainer", lightning.patch),
        ("pytorch_lightning.trainer.trainer", lightning.patch),
        ("keras", keras.patch),
    ):
        hook = when_imported(module_name, functools.partial(_apply, patch, session))
        if hook is not None:
            pending.append(hook)
    return pending
