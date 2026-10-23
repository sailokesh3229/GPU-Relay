"""PyTorch Lightning ``Trainer.fit`` hook (spec §7.2).

A callback saves ``last.ckpt`` at the end of a training epoch when the relay's checkpoint
policy is due (epoch boundaries resume exactly); a resumed segment calls ``fit`` with
``ckpt_path`` pointing at the restored checkpoint. The callback is a ``Checkpoint``
subclass so that, like ``ModelCheckpoint``, it runs after the LightningModule's own
``on_train_epoch_end``. RNG states are restored before ``fit`` (the resume point); Lightning
creates the epoch's data iterator before ``on_train_epoch_start``, so they must not be
re-applied later.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpurelay.runtime.hooks import major_version

if TYPE_CHECKING:
    from gpurelay.runtime.session import RelaySession

log = logging.getLogger(__name__)

SUPPORTED_MAJOR = (2,)
CKPT_NAME = "last.ckpt"
_PATCHED = "_gpurelay_patched"


def patch(trainer_module: types.ModuleType, session: RelaySession) -> None:
    package = trainer_module.__name__.split(".")[0]  # "lightning" or "pytorch_lightning"
    root = __import__(package)
    Trainer = trainer_module.Trainer
    if getattr(Trainer, _PATCHED, False):
        return
    if major_version(root.__version__) not in SUPPORTED_MAJOR:
        log.warning(
            "%s %s is not supported by the GPU Relay hook; Trainer.fit is not resumable",
            package,
            root.__version__,
        )
        return
    api = "lightning.pytorch" if package == "lightning" else "pytorch_lightning"
    callback_base = importlib.import_module(f"{api}.callbacks").Checkpoint
    orig_fit = Trainer.fit
    fit_signature = inspect.signature(orig_fit)
    work = session.runtime_dir / "framework-work" / "lightning"

    class RelayCallback(callback_base):  # type: ignore[misc, valid-type]
        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            if session.framework_due():
                work.mkdir(parents=True, exist_ok=True)
                with session.capture.muted():
                    trainer.save_checkpoint(str(work / CKPT_NAME))
                session.framework_safe_point(work)

    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        if session.tracker.passthrough:
            return orig_fit(self, *args, **kwargs)
        bound = fit_signature.bind_partial(self, *args, **kwargs)
        callback = RelayCallback()
        self.callbacks.append(callback)

        def run(resume: Path | None) -> Any:
            if resume is not None:
                bound.arguments["ckpt_path"] = str(resume / CKPT_NAME)
            return orig_fit(*bound.args, **bound.kwargs)

        try:
            return session.framework_call("lightning", run)
        finally:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    Trainer.fit = fit
    setattr(Trainer, _PATCHED, True)
    log.info("%s %s: Trainer.fit hook installed", package, root.__version__)
