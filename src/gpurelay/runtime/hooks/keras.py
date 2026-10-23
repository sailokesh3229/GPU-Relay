"""Keras ``Model.fit`` hook (spec §7.2), for Keras 3 (any backend).

``keras.callbacks.BackupAndRestore`` (backup in the relay's working area) is appended to
the user's callbacks; a relay callback after it snapshots the backup at the end of an
epoch when the relay's checkpoint policy is due. A resumed segment restores the backup
directory before calling ``fit``, so ``BackupAndRestore`` continues from the next epoch;
RNG states are re-applied at the start of the first resumed epoch because ``fit``'s own
setup consumes randomness before shuffling.

Limitation: the ``History`` returned by a resumed ``fit`` covers only resumed epochs.
"""

from __future__ import annotations

import inspect
import logging
import shutil
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpurelay.runtime.hooks import major_version

if TYPE_CHECKING:
    from gpurelay.runtime.session import RelaySession

log = logging.getLogger(__name__)

SUPPORTED_MAJOR = (3,)
_PATCHED = "_gpurelay_patched"


def patch(keras: types.ModuleType, session: RelaySession) -> None:
    Model = keras.Model
    if getattr(Model, _PATCHED, False):
        return
    if major_version(keras.__version__) not in SUPPORTED_MAJOR:
        log.warning(
            "keras %s is not supported by the GPU Relay hook; Model.fit is not resumable",
            keras.__version__,
        )
        return
    orig_fit = Model.fit
    fit_signature = inspect.signature(orig_fit)
    backup = session.runtime_dir / "framework-work" / "keras-backup"

    class RelayCallback(keras.callbacks.Callback):  # type: ignore[misc, name-defined]
        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            if session.framework_due():
                session.framework_safe_point(backup)

    class RngCallback(keras.callbacks.Callback):  # type: ignore[misc, name-defined]
        """First in the list: restores RNG before anything else in the resumed epoch."""

        def __init__(self) -> None:
            super().__init__()
            self.pending = True

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            if self.pending:
                self.pending = False
                session.reapply_rng()

    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        if session.tracker.passthrough:
            return orig_fit(self, *args, **kwargs)
        bound = fit_signature.bind_partial(self, *args, **kwargs)

        def run(resume: Path | None) -> Any:
            shutil.rmtree(backup, ignore_errors=True)
            if resume is not None:
                shutil.copytree(resume, backup)
            backup.mkdir(parents=True, exist_ok=True)
            callbacks = list(bound.arguments.get("callbacks") or [])
            if resume is not None:
                callbacks.insert(0, RngCallback())
            callbacks += [
                keras.callbacks.BackupAndRestore(
                    str(backup), save_freq="epoch", delete_checkpoint=True
                ),
                RelayCallback(),
            ]
            bound.arguments["callbacks"] = callbacks
            return orig_fit(*bound.args, **bound.kwargs)

        return session.framework_call("keras", run)

    Model.fit = fit
    setattr(Model, _PATCHED, True)
    log.info("keras %s: Model.fit hook installed", keras.__version__)
