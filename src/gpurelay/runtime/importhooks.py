"""Run a callback when a module is first imported (lazy patching of optional libraries).

Patching only on import keeps unused libraries out of the user's process and lets user
code configure them first (e.g. ``matplotlib.use(...)``).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import logging
import sys
import types
from collections.abc import Callable, Sequence
from importlib.machinery import ModuleSpec
from typing import Any

log = logging.getLogger(__name__)

OnLoad = Callable[[types.ModuleType], None]


class _NotifyingLoader(importlib.abc.Loader):
    """Delegating loader that calls ``on_load(module)`` after the module executes."""

    def __init__(self, inner: Any, on_load: OnLoad) -> None:
        self._inner = inner
        self._on_load = on_load

    def create_module(self, spec: ModuleSpec) -> types.ModuleType | None:
        module: types.ModuleType | None = self._inner.create_module(spec)
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        self._inner.exec_module(module)
        try:
            self._on_load(module)
        except Exception as exc:  # a broken hook must never break the user's import
            log.warning("hook for %s failed: %r", module.__name__, exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ImportHook(importlib.abc.MetaPathFinder):
    """One-shot finder for ``fullname``: lets it load normally, then calls ``on_load``."""

    def __init__(self, fullname: str, on_load: OnLoad) -> None:
        self.fullname = fullname
        self._on_load = on_load

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname != self.fullname:
            return None
        self.remove()
        spec = importlib.util.find_spec(fullname)
        if spec is not None and spec.loader is not None:
            spec.loader = _NotifyingLoader(spec.loader, self._on_load)
        return spec

    def remove(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)


def when_imported(fullname: str, on_load: OnLoad) -> ImportHook | None:
    """Call ``on_load(module)`` now if ``fullname`` is loaded, else right after its import.

    Returns the installed hook (to remove later), or None if it ran immediately.
    """
    module = sys.modules.get(fullname)
    if module is not None:
        on_load(module)
        return None
    hook = ImportHook(fullname, on_load)
    sys.meta_path.insert(0, hook)
    return hook
