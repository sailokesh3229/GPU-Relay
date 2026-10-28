"""Code that will fail on GPU Relay's platforms as written (shown before a job is submitted).

A job runs on Kaggle, Modal or Colab, so code made for one notebook service stops on the
others, and a notebook copied in parts can use names no cell defines. Found statically:

* ``google.colab`` imports (``drive.mount``, ``files``, ...): Colab only;
* paths under ``/content/drive``: Google Drive, mounted only in Colab;
* names that are read but never bound anywhere (no assignment, import, def, argument, ...):
  the job stops with ``NameError`` there, often after a long setup.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterable

from gpurelay.analyzer.static import Finding

# Names a notebook or script may use without defining them.
_PROVIDED = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__builtins__", "__spec__", "__loader__", "__package__",
    "get_ipython", "display", "In", "Out", "exit", "quit",
}  # fmt: skip


def _colab_imports(tree: ast.Module, cell: int | None) -> list[Finding]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(n == "google.colab" or n.startswith("google.colab.") for n in names):
            found.append(Finding(
                "colab_only",
                "`google.colab` exists only on Google Colab, so this stops the job on Kaggle "
                "and Modal (ModuleNotFoundError). Remove it (and drive.mount): GPU Relay keeps "
                "the job's files itself; use relative paths such as checkpoints/model.pt",
                node.lineno, cell,
            ))  # fmt: skip
    return found


def _drive_paths(tree: ast.Module, cell: int | None) -> list[Finding]:
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("/content/drive")
        ):
            found.append(Finding(
                "colab_path",
                f"`{node.value}` is on Google Drive, which only Colab mounts: on Kaggle and "
                "Modal it is an empty local folder (files from Drive are missing, and files "
                "written there are not part of the job's outputs). Use a relative path",
                node.lineno, cell,
            ))  # fmt: skip
    return found


def _bound_names(trees: Iterable[ast.Module]) -> tuple[set[str], bool]:
    """Every name bound anywhere, and whether a ``from x import *`` makes this unknowable."""
    bound: set[str] = set()
    star = False
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
                bound.add(node.id)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                bound.add(node.name)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        star = True
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.Global | ast.Nonlocal):
                bound.update(node.names)
            elif isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                bound.add(node.rest)
    return bound, star


def _undefined_names(trees: list[ast.Module], cells: bool) -> list[Finding]:
    bound, star = _bound_names(trees)
    if star:
        return []  # names may come from the star import: no reliable answer
    found: list[Finding] = []
    seen: set[str] = set()
    for number, tree in enumerate(trees, start=1):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in bound
                and node.id not in _PROVIDED
                and node.id not in seen
            ):
                seen.add(node.id)
                found.append(Finding(
                    "undefined_name",
                    f"`{node.id}` is used but never defined (no cell or line assigns or imports "
                    "it), so the job stops here with NameError. Add the code that creates it",
                    node.lineno, number if cells else None,
                ))  # fmt: skip
    return found


def will_fail(trees: list[ast.Module], *, cells: bool) -> list[Finding]:
    """Problems that stop the job as written (``cells``: the trees are notebook cells)."""
    errors: list[Finding] = []
    for number, tree in enumerate(trees, start=1):
        cell = number if cells else None
        errors += _colab_imports(tree, cell) + _drive_paths(tree, cell)
    return errors + _undefined_names(trees, cells)
