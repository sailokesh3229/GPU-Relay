"""Hugging Face ``transformers.Trainer`` hook (spec §7.2).

* ``Trainer.train`` runs as a resumable framework event; on resume it receives
  ``resume_from_checkpoint=<restored checkpoint>``.
* A callback asks the Trainer to save when the relay's checkpoint policy is due. A save
  the user's own ``save_strategy`` would not have made is written to a private directory
  (so ``output_dir`` looks exactly as in an uninterrupted run), runs with output muted,
  and is deleted once it is part of the relay checkpoint.
* Saves are aligned to logging steps: the Trainer's running loss accumulator is not part
  of its checkpoint, so a resume elsewhere would change the next logged loss. Only a
  handoff that cannot wait saves between logging steps.
"""

from __future__ import annotations

import logging
import shutil
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpurelay.runtime.hooks import major_version

if TYPE_CHECKING:
    from gpurelay.runtime.session import RelaySession

log = logging.getLogger(__name__)

SUPPORTED_MAJOR = (4, 5)
_PATCHED = "_gpurelay_patched"


def _logging_interval(args: Any, state: Any) -> int:
    """Steps between loss logs (1 when the Trainer does not log by steps)."""
    strategy = str(getattr(args, "logging_strategy", "steps")).lower()
    if "steps" not in strategy:
        return 1
    steps = getattr(args, "logging_steps", 0) or 0
    if 0 < steps < 1:  # a ratio of total steps
        steps = max(1, int(steps * max(1, state.max_steps)))
    return max(1, int(steps))


def _at_logging_step(args: Any, state: Any) -> bool:
    """True when this step will log (the loss accumulator is reset right after)."""
    return bool(state.global_step % _logging_interval(args, state) == 0)


def patch(trainer_module: types.ModuleType, session: RelaySession) -> None:
    import transformers

    Trainer = trainer_module.Trainer
    if getattr(Trainer, _PATCHED, False):
        return
    if major_version(transformers.__version__) not in SUPPORTED_MAJOR or not hasattr(
        Trainer, "_save_checkpoint"
    ):
        log.warning(
            "transformers %s is not supported by the GPU Relay hook; Trainer.train is not "
            "resumable mid-training (see `gpurelay analyze`)",
            transformers.__version__,
        )
        return
    callback_base = transformers.TrainerCallback
    orig_train = Trainer.train
    orig_save = Trainer._save_checkpoint
    private_root = session.runtime_dir / "framework-work" / "hf"

    class RelayCallback(callback_base):  # type: ignore[misc, valid-type]
        def __init__(self, trainer: Any) -> None:
            self.trainer = trainer
            self.pending: str | None = None
            self.lookahead_s = 0.0
            self.step_s = 0.0  # typical step time (saves excluded)
            self.save_s = 0.0  # last measured save time
            self._last_step_end: float | None = None
            self._saved_since = False
            self._save_started = 0.0

        def _time_step(self) -> None:
            now = time.monotonic()
            if self._last_step_end is not None and not self._saved_since:
                gap = now - self._last_step_end
                self.step_s = gap if not self.step_s else 0.8 * self.step_s + 0.2 * gap
            self._last_step_end = now
            self._saved_since = False

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self._time_step()
            if self.pending is not None:
                return control
            # At a logging step, look ahead to the next one: hand off now if it would be
            # too late. Between logging steps, only an imminent deadline forces a save.
            if _at_logging_step(args, state):
                self.lookahead_s = self.step_s * _logging_interval(args, state) + self.save_s
                due = session.framework_due(self.lookahead_s)
            else:
                self.lookahead_s = self.step_s + self.save_s
                due = session.framework_due(self.lookahead_s) and session.watchdog.handoff_due(
                    self.lookahead_s
                )
            if due:
                self._save_started = time.monotonic()
                if control.should_save:
                    self.pending = "user"
                else:
                    self.pending = "forced"
                    control.should_save = True
                    self.trainer._gpurelay_forced_dir = str(private_root)
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            kind, self.pending = self.pending, None
            if kind is None:
                return control
            self.trainer._gpurelay_forced_dir = None
            self.save_s = time.monotonic() - self._save_started
            self._saved_since = True
            folder = f"checkpoint-{state.global_step}"
            if kind == "forced":
                path = private_root / folder
            else:
                path = Path(self.trainer._get_output_dir(trial=None)) / folder
            try:
                session.framework_safe_point(path, self.lookahead_s)
            finally:
                if kind == "forced":
                    shutil.rmtree(path, ignore_errors=True)
            return control

    def train(self: Any, *args: Any, **kwargs: Any) -> Any:
        if session.tracker.passthrough:
            return orig_train(self, *args, **kwargs)
        callback = RelayCallback(self)
        self.add_callback(callback)

        def run(resume: Path | None) -> Any:
            if resume is not None:
                kwargs["resume_from_checkpoint"] = str(resume)
            return orig_train(self, *args, **kwargs)

        try:
            return session.framework_call("hf", run)
        finally:
            self.remove_callback(callback)

    def _save_checkpoint(self: Any, model: Any, trial: Any, *args: Any, **kwargs: Any) -> Any:
        forced = getattr(self, "_gpurelay_forced_dir", None)
        if not forced:
            return orig_save(self, model, trial, *args, **kwargs)
        self._get_output_dir = lambda trial=None: forced
        try:
            with session.capture.muted():
                return orig_save(self, model, trial, *args, **kwargs)
        finally:
            del self._get_output_dir

    Trainer.train = train
    Trainer._save_checkpoint = _save_checkpoint
    setattr(Trainer, _PATCHED, True)
    log.info("transformers %s: Trainer hook installed", transformers.__version__)
