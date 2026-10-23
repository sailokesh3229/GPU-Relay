"""Variable classification, capture, and restore (spec §7.5).

Classes of variables:

* **skip** — modules, functions, classes (recreated by replay), IPython internals,
  dunder names, and ``__relay__`` itself. Never snapshotted. Modules, and functions/classes
  of importable modules, are *recorded by name* instead: a binding made where replay does
  not recreate it (``try: import x`` / ``if ...: from y import f``) is imported again on
  restore.
* **torch** — objects exposing ``state_dict()``/``load_state_dict()`` while torch is
  loaded (``nn.Module``, optimizers, LR schedulers, ``GradScaler``, EMA helpers, ...).
  Saved with ``torch.save`` as state dicts **and**, when picklable, as whole objects in
  the same file (tensor storages are deduplicated, so the extra cost is small).
  On restore the state dict is loaded **into the already-existing live object**, which
  preserves optimizer <-> parameter references; the whole object is used only when no
  compatible live object exists.
* **plain** — everything else, saved with cloudpickle. Unpicklable values (open files,
  locks, live generators, ...) are skipped and reported by name.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import pickle
import re
import sys
import types
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cloudpickle
from pydantic import BaseModel, Field

from gpurelay.errors import StateRestoreError

log = logging.getLogger(__name__)

TORCH_STATE_FILE = "torch_state.pt"
VARS_FILE = "vars.pkl"
IMPORTS_FILE = "imports.json"  # bindings to import again by name on restore
STATE_FORMAT_VERSION = 1
RELAY_NAME = "__relay__"

_IPYTHON_NAMES = frozenset(
    {"In", "Out", "get_ipython", "exit", "quit", "_", "__", "___", "_i", "_ii", "_iii"}
    | {"_oh", "_ih", "_dh", "_sh"}
)
_IPYTHON_NUMBERED = re.compile(r"^_(i|ii|iii)?\d+$")


class StateManifest(BaseModel):
    """What a capture saved and skipped; stored in the checkpoint's ``meta.json``."""

    torch: list[str] = Field(default_factory=list)
    torch_whole_objects: list[str] = Field(default_factory=list)
    plain: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)
    unpicklable: dict[str, str] = Field(default_factory=dict)
    # name -> [module] for modules, [module, qualname] for importable functions/classes
    imports: dict[str, list[str]] = Field(default_factory=dict)


@dataclass
class RestoreReport:
    """Outcome of :func:`restore_state`."""

    loaded_into_live: list[str] = field(default_factory=list)
    unpickled_torch: list[str] = field(default_factory=list)
    plain: list[str] = field(default_factory=list)
    imported: list[str] = field(default_factory=list)  # bindings imported again by name
    warnings: list[str] = field(default_factory=list)


# ====================================================================== classification


def skip_reason(name: str, value: Any) -> str | None:
    """Why ``name`` must never be snapshotted, or None if it may be."""
    if name == RELAY_NAME:
        return "gpurelay runtime"
    if name.startswith("__") and name.endswith("__"):
        return "dunder name"
    if name in _IPYTHON_NAMES or _IPYTHON_NUMBERED.match(name):
        return "IPython internal"
    if isinstance(value, types.ModuleType):
        return "module"
    if isinstance(value, type):
        return "class"
    if isinstance(value, types.FunctionType | types.BuiltinFunctionType | types.MethodType):
        return "function"
    return None


def import_ref(value: Any, own_module: str | None) -> list[str] | None:
    """How to import ``value`` again: [module] or [module, qualname]; None if not possible.

    Objects defined by the job itself (``own_module``) are recreated by replay, not imported.
    """
    if isinstance(value, types.ModuleType):
        name = getattr(value, "__name__", None)
        return [name] if isinstance(name, str) and sys.modules.get(name) is value else None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str) or module == own_module:
        return None
    target: Any = sys.modules.get(module)
    for part in qualname.split("."):
        target = getattr(target, part, None)
    return [module, qualname] if target is value else None


def _import_again(ref: list[str]) -> Any:
    target: Any = importlib.import_module(ref[0])
    for part in ref[1].split(".") if len(ref) > 1 else ():
        target = getattr(target, part)
    return target


def is_torch_stateful(value: Any) -> bool:
    """True for objects saved via ``state_dict`` (only when torch is loaded)."""
    if "torch" not in sys.modules or isinstance(value, type):
        return False
    try:
        return callable(getattr(value, "state_dict", None)) and callable(
            getattr(value, "load_state_dict", None)
        )
    except Exception:
        return False


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _restore_priority(value: Any) -> int:
    """Modules first, then optimizers (need live params), then schedulers/others."""
    torch = sys.modules.get("torch")
    if torch is not None:
        if isinstance(value, torch.nn.Module):
            return 0
        if isinstance(value, torch.optim.Optimizer):
            return 1
    return 2


# ====================================================================== capture


def capture_state(
    namespace: Mapping[str, Any], out_dir: Path, names: Iterable[str] | None = None
) -> StateManifest:
    """Write the selected variables of ``namespace`` into ``out_dir``.

    ``names`` selects plain variables to save (default: all eligible). Torch-stateful
    objects are always saved, whether selected or not (spec §7.3).
    """
    selected = None if names is None else set(names)
    own_module = namespace.get("__name__")
    manifest = StateManifest()
    torch_vars: dict[str, Any] = {}
    plain_vars: dict[str, Any] = {}

    for name in sorted(namespace):
        value = namespace[name]
        wanted = selected is None or name in selected
        reason = skip_reason(name, value)
        if reason is not None:
            if wanted:
                manifest.skipped[name] = reason
            if reason in ("module", "class", "function"):
                ref = import_ref(value, own_module)
                if ref is not None:
                    manifest.imports[name] = ref
            continue
        if is_torch_stateful(value):
            torch_vars[name] = value
        elif wanted:
            plain_vars[name] = value

    if torch_vars:
        manifest.torch = sorted(torch_vars)
        manifest.torch_whole_objects = _save_torch(torch_vars, out_dir / TORCH_STATE_FILE)
    if plain_vars:
        saved, unpicklable = _save_plain(plain_vars, out_dir / VARS_FILE)
        manifest.plain = saved
        manifest.unpicklable = unpicklable
        for name, why in unpicklable.items():
            log.debug("variable %r is not picklable and was not checkpointed: %s", name, why)
    if manifest.imports:
        (out_dir / IMPORTS_FILE).write_text(json.dumps(manifest.imports), encoding="utf-8")
    return manifest


def _torch_save_bytes_ok(torch: Any, obj: Any) -> bool:
    try:
        torch.save(obj, io.BytesIO(), pickle_module=cloudpickle)
        return True
    except Exception:
        return False


def _save_torch(torch_vars: dict[str, Any], path: Path) -> list[str]:
    """Save state dicts plus picklable whole objects; return names with whole objects."""
    torch = sys.modules["torch"]
    state_dicts = {name: value.state_dict() for name, value in torch_vars.items()}
    types_ = {name: _type_name(value) for name, value in torch_vars.items()}
    objects = dict(torch_vars)

    def payload() -> dict[str, Any]:
        return {
            "format": STATE_FORMAT_VERSION,
            "state_dicts": state_dicts,
            "types": types_,
            "objects": objects,
        }

    try:
        torch.save(payload(), path, pickle_module=cloudpickle)
    except Exception:
        # Find which whole objects cannot be pickled; keep their state dicts only.
        for name in list(objects):
            if not _torch_save_bytes_ok(torch, objects[name]):
                log.warning("torch object %r has no whole-object fallback (unpicklable)", name)
                del objects[name]
        torch.save(payload(), path, pickle_module=cloudpickle)
    return sorted(objects)


def _save_plain(plain_vars: dict[str, Any], path: Path) -> tuple[list[str], dict[str, str]]:
    """Cloudpickle plain variables together (keeps shared references)."""
    unpicklable: dict[str, str] = {}
    try:
        blob = cloudpickle.dumps(plain_vars, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        good: dict[str, Any] = {}
        for name, value in plain_vars.items():
            try:
                cloudpickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                unpicklable[name] = f"{type(exc).__name__}: {exc}"
            else:
                good[name] = value
        plain_vars = good
        blob = cloudpickle.dumps(plain_vars, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(blob)
    return sorted(plain_vars), unpicklable


# ====================================================================== restore


def _default_map_location(torch: Any) -> str | None:
    # VERIFY: multi-GPU sessions; tensors saved on cuda:N with N >= device count.
    return None if torch.cuda.is_available() else "cpu"


def restore_state(namespace: MutableMapping[str, Any], in_dir: Path) -> RestoreReport:
    """Restore variables saved by :func:`capture_state` into ``namespace``.

    RNG state is restored separately, after this (see ``runtime.rng``).
    """
    report = RestoreReport()
    torch_path = in_dir / TORCH_STATE_FILE
    if torch_path.exists():
        _restore_torch(namespace, torch_path, report)
    vars_path = in_dir / VARS_FILE
    if vars_path.exists():
        values: dict[str, Any] = pickle.loads(vars_path.read_bytes())
        namespace.update(values)
        report.plain = sorted(values)
    imports_path = in_dir / IMPORTS_FILE
    if imports_path.exists():
        refs: dict[str, list[str]] = json.loads(imports_path.read_text(encoding="utf-8"))
        for name, ref in sorted(refs.items()):
            if name in namespace:
                continue  # replay already recreated it
            try:
                namespace[name] = _import_again(ref)
                report.imported.append(name)
            except Exception as exc:
                what = ".".join(ref)
                report.warnings.append(f"could not import {what} again for {name!r}: {exc}")
    for message in report.warnings:
        log.warning(message)
    return report


def _restore_torch(namespace: MutableMapping[str, Any], path: Path, report: RestoreReport) -> None:
    torch = importlib.import_module("torch")
    payload = torch.load(path, map_location=_default_map_location(torch), weights_only=False)
    state_dicts: dict[str, Any] = payload["state_dicts"]
    types_: dict[str, str] = payload["types"]
    objects: dict[str, Any] = payload["objects"]

    def priority(name: str) -> int:
        live = namespace.get(name)
        return _restore_priority(live if live is not None else objects.get(name))

    for name in sorted(state_dicts, key=lambda n: (priority(n), n)):
        live = namespace.get(name)
        if live is not None and is_torch_stateful(live) and _type_name(live) == types_[name]:
            try:
                live.load_state_dict(state_dicts[name])
            except Exception as exc:
                if name not in objects:
                    raise StateRestoreError(
                        f"cannot load saved state into live {name!r}: {exc}"
                    ) from exc
                report.warnings.append(
                    f"load_state_dict failed for {name!r} ({exc}); using the saved object"
                )
            else:
                report.loaded_into_live.append(name)
                continue
        elif live is not None:
            report.warnings.append(
                f"live {name!r} is {_type_name(live)}, saved {types_[name]}; using saved object"
            )

        if name not in objects:
            raise StateRestoreError(
                f"no live object for {name!r} and no picklable saved object to fall back to"
            )
        namespace[name] = objects[name]
        report.unpickled_torch.append(name)
