"""Layer C: automatic loop instrumentation and static analysis of scripts (spec §7.3).

The script is split into **top-level statements** (treated like notebook cells, spec §7.1).
For each statement this module computes:

* the transformed, compiled code: every *outermost* ``for`` loop reachable from module level
  (descending into ``if``/``with``/``try``/``match`` bodies, never into loop bodies,
  functions, or classes) becomes ``for x in __relay__.loop("<loop_id>", ITER):``. The AST is
  compiled directly, so tracebacks keep the user's line numbers;
* the names it assigns or mutates (assignments, augmented assignments, ``del``, loop and
  ``with`` targets, attribute/subscript stores, mutating method calls such as ``.append``,
  in-place tensor methods ending in ``_``), plus names passed as call arguments;
* side effects (downloads, deletions, subprocesses, writes to absolute paths) and
  uninstrumented loops (``while``, ``async for``, loops inside called functions), used by
  replay decisions and by the analyzer.
"""

from __future__ import annotations
import __future__

import ast
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from types import CodeType
from typing import Any

RELAY_NAME = "__relay__"

MUTATING_METHODS = frozenset(
    {
        "append", "extend", "insert", "pop", "remove", "clear", "update", "add",
        "discard", "setdefault", "popitem", "sort", "reverse", "appendleft",
        "extendleft", "popleft", "rotate", "fill", "resize", "put", "__setitem__",
        "__delitem__", "load_state_dict", "step", "zero_grad", "train", "eval",
    }
)  # fmt: skip

# Dotted-name suffixes of calls with external side effects (never replayed automatically).
SIDE_EFFECT_CALLS = {
    "urlretrieve": "download",
    "urlopen": "download",
    "requests.get": "download",
    "requests.post": "network request",
    "wget.download": "download",
    "gdown.download": "download",
    "hf_hub_download": "download",
    "snapshot_download": "download",
    "torch.hub.load": "download",
    "torch.hub.download_url_to_file": "download",
    "os.remove": "file deletion",
    "os.unlink": "file deletion",
    "os.rmdir": "file deletion",
    "os.removedirs": "file deletion",
    "shutil.rmtree": "file deletion",
    "shutil.move": "file move",
    "unlink": "file deletion",
    "rmdir": "file deletion",
    "subprocess.run": "subprocess",
    "subprocess.call": "subprocess",
    "subprocess.check_call": "subprocess",
    "subprocess.check_output": "subprocess",
    "subprocess.Popen": "subprocess",
    "os.system": "subprocess",
}

_DEFINITIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

FRAMEWORK_MODULES = {
    "transformers": "hf",
    "lightning": "lightning",
    "pytorch_lightning": "lightning",
    "keras": "keras",
    "tensorflow": "keras",
}


@dataclass(frozen=True)
class InstrumentOptions:
    """What automatic instrumentation covers (from ``RuntimeSettings``)."""

    loop_depth: int = 2  # levels of nested ``for`` loops with safe points
    while_loops: bool = False
    functions: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> InstrumentOptions:
        return cls(
            loop_depth=settings.loop_depth,
            while_loops=settings.instrument_while,
            functions=settings.instrument_functions,
        )


@dataclass(frozen=True)
class LoopInfo:
    loop_id: str
    lineno: int
    header: str
    kind: str  # "for" | "while" | "async for" | "sdk"


@dataclass
class FunctionInfo:
    name: str
    lineno: int
    loops: list[LoopInfo] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    calls: set[str] = field(default_factory=set)
    global_names: set[str] = field(default_factory=set)
    # A top-level, non-generator ``def``: its loops can be instrumented (opt-in).
    instrumentable: bool = False


@dataclass
class StatementInfo:
    index: int
    lineno: int
    end_lineno: int
    kind: str
    code: CodeType | None  # None for notebook cells (IPython compiles them)
    assigned: frozenset[str]
    call_args: frozenset[str]
    calls: frozenset[str]  # simple names of called functions (for module-function lookup)
    method_calls: frozenset[str]  # attribute names of method calls (e.g. "fit", "train")
    loops: list[LoopInfo] = field(default_factory=list)  # instrumented (or SDK) loops
    uninstrumented: list[LoopInfo] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    is_definition: bool = False  # import / def / class: always replayed
    is_setup: bool = False  # environment setup (pip/conda installs): always replayed
    source: str | None = None  # raw notebook cell source
    # Top-level imports/defs/classes of a cell that is not *only* definitions: executed
    # even when the cell itself is skipped on resume (modules/functions are never pickled).
    definitions_code: CodeType | None = None

    @property
    def has_safe_points(self) -> bool:
        return bool(self.loops)


@dataclass
class ScriptPlan:
    filename: str
    statements: list[StatementInfo]
    functions: dict[str, FunctionInfo]
    global_names: frozenset[str]  # names declared ``global`` inside functions
    imported_modules: frozenset[str]  # top-level module names imported anywhere
    sdk_aliases: frozenset[str]
    docstring: str | None = None
    options: InstrumentOptions = field(default_factory=InstrumentOptions)
    sdk_names: frozenset[str] = frozenset()  # names from ``from gpurelay import range``

    def frameworks(self) -> set[str]:
        return {FRAMEWORK_MODULES[m] for m in self.imported_modules if m in FRAMEWORK_MODULES}


# ====================================================================== helpers


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _base_name(node: ast.AST) -> str | None:
    """``a`` for ``a``, ``a.b.c``, ``a[0].b``, ``a.b(...)[1]``."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute | ast.Subscript | ast.Starred):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def _side_effect(call: ast.Call) -> str | None:
    name = _dotted(call.func) or ""
    for suffix, what in SIDE_EFFECT_CALLS.items():
        if name == suffix or name.endswith("." + suffix):
            return f"{what}: {name}()"
    for kw in call.keywords:
        if kw.arg == "download" and isinstance(kw.value, ast.Constant) and kw.value.value:
            return f"download: {name}(download=True)"
    if name == "open" and call.args:
        first = call.args[0]
        mode = call.args[1] if len(call.args) > 1 else None
        for kw in call.keywords:
            if kw.arg == "mode":
                mode = kw.value
        writes = isinstance(mode, ast.Constant) and any(c in str(mode.value) for c in "wax+")
        if (
            writes
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and (first.value.startswith(("/", "\\")) or first.value[1:3] in (":\\", ":/"))
        ):
            return f"write outside the working directory: {first.value}"
    return None


class _Scan(ast.NodeVisitor):
    """Collects writes, call info, and side effects without entering nested scopes."""

    def __init__(self) -> None:
        self.assigned: set[str] = set()
        self.call_args: set[str] = set()
        self.calls: set[str] = set()
        self.method_calls: set[str] = set()
        self.side_effects: list[str] = []
        self.imports: set[str] = set()

    # scopes ---------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.assigned.add(node.name)
        for deco in node.decorator_list:
            self.visit(deco)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.assigned.add(node.name)
        for deco in node.decorator_list:
            self.visit(deco)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.assigned.add(node.name)
        for deco in node.decorator_list:
            self.visit(deco)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def _comprehension(self, node: ast.AST, elts: list[ast.AST]) -> None:
        for gen in node.generators:  # type: ignore[attr-defined]
            self.visit(gen.iter)
            for cond in gen.ifs:
                self.visit(cond)
        for elt in elts:
            self.visit(elt)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node, [node.key, node.value])

    # writes ---------------------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            self.assigned.add(node.id)

    def _store_base(self, node: ast.Attribute | ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            base = _base_name(node.value)
            if base:
                self.assigned.add(base)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._store_base(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._store_base(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.assigned.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.assigned.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.assigned.add(node.name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.assigned.add(alias.asname or alias.name.split(".")[0])
            self.imports.add(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.assigned.add(alias.asname or alias.name)
        if node.module and node.level == 0:
            self.imports.add(node.module.split(".")[0])

    # calls ----------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            self.method_calls.add(attr)
            if attr in MUTATING_METHODS or (attr.endswith("_") and not attr.startswith("_")):
                base = _base_name(node.func.value)
                if base:
                    self.assigned.add(base)
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            if isinstance(arg, ast.Name):
                self.call_args.add(arg.id)
        effect = _side_effect(node)
        if effect:
            self.side_effects.append(f"line {node.lineno}: {effect}")
        self.generic_visit(node)


def _segment(source: str | None, node: ast.AST) -> str:
    if source is None:
        return ast.unparse(node)
    return ast.get_source_segment(source, node) or "?"


def _loop_header(source: str | None, node: ast.For | ast.AsyncFor | ast.While) -> str:
    """Loop header text (from ``source``, or ``ast.unparse`` when there is none)."""
    if isinstance(node, ast.While):
        return f"while {_segment(source, node.test)}"
    target = _segment(source, node.target)
    iterable = _segment(source, node.iter)
    prefix = "async for" if isinstance(node, ast.AsyncFor) else "for"
    return f"{prefix} {target} in {iterable}"


def make_loop_id(filename: str, lineno: int, header: str) -> str:
    """Stable id from file name + line + loop header source (spec §7.3)."""
    digest = hashlib.sha1(f"{filename}:{lineno}:{header}".encode()).hexdigest()[:6]
    return f"L{lineno}-{digest}"


def _is_sdk_loop(node: ast.For, sdk_aliases: frozenset[str], sdk_names: frozenset[str]) -> bool:
    it = node.iter
    if not isinstance(it, ast.Call):
        return False
    func = it.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in sdk_aliases and func.attr in ("range", "chunks")
    return isinstance(func, ast.Name) and func.id in sdk_names


class _Instrumenter:
    """Rewrites the ``for`` loops of one top-level statement in place: the outermost ones
    and, up to ``options.loop_depth`` levels, the loops nested in their bodies."""

    def __init__(
        self,
        source: str | None,
        filename: str,
        sdk_aliases: frozenset[str],
        sdk_names: frozenset[str],
        options: InstrumentOptions | None = None,
    ) -> None:
        self.source = source
        self.filename = filename
        self.sdk_aliases = sdk_aliases
        self.sdk_names = sdk_names
        self.options = options or InstrumentOptions()
        self.loops: list[LoopInfo] = []
        self.uninstrumented: list[LoopInfo] = []
        self.function_loops: list[LoopInfo] = []  # instrumented inside functions (opt-in)
        self._depth = 0  # loop levels around the statement being visited
        self._locals: tuple[str, ...] | None = None  # inside a function: its local names

    def _nested(self, body: list[ast.stmt]) -> None:
        """Visit a loop body one level deeper (if nested loops are instrumented that far)."""
        if self._depth + 1 >= self.options.loop_depth:
            return
        self._depth += 1
        try:
            self.block(body)
        finally:
            self._depth -= 1

    def block(self, stmts: list[ast.stmt]) -> None:
        for index, stmt in enumerate(stmts):
            stmts[index] = self.stmt(stmt)

    def _relay_call(self, attr: str, args: list[ast.expr], at: ast.AST) -> ast.Call:
        if attr == "loop" and self._locals is not None:
            # Loops inside functions also pass the function's local names (snapshotted).
            names: list[ast.expr] = [ast.Constant(n) for n in self._locals]
            args = [*args, ast.Tuple(elts=names, ctx=ast.Load())]
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=RELAY_NAME, ctx=ast.Load()), attr=attr, ctx=ast.Load()
            ),
            args=args,
            keywords=[],
        )
        return ast.copy_location(call, at)

    def _record(self, info: LoopInfo) -> None:
        (self.function_loops if self._locals is not None else self.loops).append(info)

    def _while(self, node: ast.While, loop_id: str, header: str) -> ast.stmt:
        """``while C: B`` -> ``for __relay_iter__ in loop(id, forever()): if not C: break; B``.

        ``continue``/``break`` keep their meaning; on resume, finished iterations are skipped
        and ``C`` is evaluated again on the restored state.
        """
        check = ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=node.test),
            body=[ast.copy_location(ast.Break(), node)],
            orelse=[],
        )
        loop = ast.For(
            target=ast.Name(id="__relay_iter__", ctx=ast.Store()),
            iter=self._relay_call(
                "loop",
                [ast.Constant(loop_id), self._relay_call("forever", [], node)],
                node,
            ),
            body=[ast.copy_location(check, node.test), *node.body],
            orelse=[],
        )
        self._record(LoopInfo(loop_id, node.lineno, header, "while"))
        self._nested(loop.body)
        return ast.copy_location(loop, node)

    def _function(self, node: ast.FunctionDef) -> None:
        """Instrument the loops of a top-level function (opt-in); its locals are snapshotted."""
        if self._locals is not None or _is_generator_function(node):
            return
        scan = _Scan()
        for child in node.body:
            scan.visit(child)
        declared = {
            name
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            if isinstance(child, ast.Global | ast.Nonlocal)
            for name in child.names
        }
        saved_depth, self._depth = self._depth, 0
        self._locals = tuple(sorted(scan.assigned - declared))
        try:
            self.block(node.body)
        finally:
            self._locals = None
            self._depth = saved_depth

    def stmt(self, node: ast.stmt) -> ast.stmt:
        """Instrument ``node`` in place; returns the node to use (a ``while`` is replaced)."""
        if isinstance(node, ast.FunctionDef) and self.options.functions:
            self._function(node)
        elif isinstance(node, ast.While) and self.options.while_loops and not node.orelse:
            header = _loop_header(self.source, node)
            return self._while(node, make_loop_id(self.filename, node.lineno, header), header)
        elif isinstance(node, ast.For):
            header = _loop_header(self.source, node)
            loop_id = make_loop_id(self.filename, node.lineno, header)
            if _is_sdk_loop(node, self.sdk_aliases, self.sdk_names):
                self._record(LoopInfo(loop_id, node.lineno, header, "sdk"))
                self._nested(node.body)
                return node
            node.iter = self._relay_call("loop", [ast.Constant(loop_id), node.iter], node.iter)
            self._record(LoopInfo(loop_id, node.lineno, header, "for"))
            self._nested(node.body)
        elif isinstance(node, ast.While | ast.AsyncFor):
            header = _loop_header(self.source, node)
            kind = "while" if isinstance(node, ast.While) else "async for"
            loop_id = make_loop_id(self.filename, node.lineno, header)
            if self._locals is None:  # loops in functions are reported per function
                self.uninstrumented.append(LoopInfo(loop_id, node.lineno, header, kind))
        elif isinstance(node, ast.If):
            self.block(node.body)
            self.block(node.orelse)
        elif isinstance(node, ast.With | ast.AsyncWith):
            self.block(node.body)
        elif isinstance(node, ast.Try | getattr(ast, "TryStar", ast.Try)):
            self.block(node.body)
            for handler in node.handlers:
                self.block(handler.body)
            self.block(node.orelse)
            self.block(node.finalbody)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                self.block(case.body)
        return node


def _is_generator_function(node: ast.FunctionDef) -> bool:
    """Contains ``yield``/``await`` in its own scope (not in nested functions)."""
    todo: list[ast.AST] = list(node.body)
    while todo:
        child = todo.pop()
        if isinstance(child, ast.Yield | ast.YieldFrom | ast.Await):
            return True
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            continue
        todo.extend(ast.iter_child_nodes(child))
    return False


def _future_flags(tree: ast.Module) -> int:
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                feature = getattr(__future__, alias.name, None)
                if feature is not None:
                    flags |= feature.compiler_flag
    return flags


def _function_infos(tree: ast.Module, source: str | None, filename: str) -> dict[str, FunctionInfo]:
    infos: dict[str, FunctionInfo] = {}
    top_level = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        info = FunctionInfo(node.name, node.lineno)
        info.instrumentable = (
            isinstance(node, ast.FunctionDef)
            and id(node) in top_level
            and not _is_generator_function(node)
        )
        for sub in ast.walk(node):
            if isinstance(sub, ast.For | ast.AsyncFor | ast.While):
                header = _loop_header(source, sub)
                kind = {ast.For: "for", ast.AsyncFor: "async for", ast.While: "while"}[type(sub)]
                info.loops.append(
                    LoopInfo(make_loop_id(filename, sub.lineno, header), sub.lineno, header, kind)
                )
            elif isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    info.calls.add(sub.func.id)
                effect = _side_effect(sub)
                if effect:
                    info.side_effects.append(f"line {sub.lineno}: {effect}")
            elif isinstance(sub, ast.Global):
                info.global_names.update(sub.names)
        infos.setdefault(node.name, info)
    return infos


def _sdk_aliases(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    aliases: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gpurelay":
                    aliases.add(alias.asname or "gpurelay")
        elif isinstance(node, ast.ImportFrom) and node.module in ("gpurelay", "gpurelay.sdk"):
            for alias in node.names:
                if alias.name in ("range", "chunks"):
                    names.add(alias.asname or alias.name)
    return frozenset(aliases), frozenset(names)


# ====================================================================== entry point


def plan_script(
    source: str | bytes,
    filename: str,
    *,
    loop_file_id: str | None = None,
    options: InstrumentOptions | None = None,
) -> ScriptPlan:
    """Parse, analyze, and instrument a script.

    ``filename`` is used for compilation (tracebacks); ``loop_file_id`` (default: the
    file's base name) is used in loop ids so they are stable across sessions whose
    working directories differ.
    """
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    tree = ast.parse(text, filename=filename)
    file_id = loop_file_id or filename.replace("\\", "/").rsplit("/", 1)[-1]
    flags = _future_flags(tree)
    sdk_aliases, sdk_names = _sdk_aliases(tree)
    functions = _function_infos(tree, text, file_id)
    global_names = frozenset(n for f in functions.values() for n in f.global_names)

    statements: list[StatementInfo] = []
    imported: set[str] = set()
    for index, node in enumerate(tree.body):
        scan = _Scan()
        scan.visit(node)
        imported |= scan.imports
        inst = _Instrumenter(text, file_id, sdk_aliases, sdk_names, options)
        node = inst.stmt(node)
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, filename, "exec", flags=flags, dont_inherit=True)
        statements.append(
            StatementInfo(
                index=index,
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                kind=type(node).__name__,
                code=code,
                assigned=frozenset(scan.assigned),
                call_args=frozenset(scan.call_args),
                calls=frozenset(scan.calls),
                method_calls=frozenset(scan.method_calls),
                loops=inst.loops,
                uninstrumented=inst.uninstrumented,
                side_effects=scan.side_effects,
                is_definition=isinstance(node, _DEFINITIONS),
            )
        )
    return ScriptPlan(
        filename=filename,
        statements=statements,
        functions=functions,
        global_names=global_names,
        imported_modules=frozenset(imported),
        sdk_aliases=sdk_aliases,
        docstring=ast.get_docstring(tree),
        sdk_names=sdk_names,
        options=options or InstrumentOptions(),
    )


# ====================================================================== notebooks

_SHELL_PREFIXES = ("!", "%system", "%sx", "%%bash", "%%sh", "%%script", "%%cmd", "%%powershell")
_SETUP_COMMAND = re.compile(
    r"^\s*(!|%)\s*(python\d*\s+-m\s+)?(pip|pip3|conda|mamba|apt|apt-get)\s+install\b"
)


def cell_file_id(index: int) -> str:
    """Pseudo file name used for loop ids of notebook cell ``index``."""
    return f"cell-{index}"


def _cell_commands(raw: str) -> tuple[list[str], bool]:
    """Shell-command side effects of a raw cell, and whether it installs packages."""
    effects: list[str] = []
    setup = False
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if _SETUP_COMMAND.match(stripped):
            setup = True
        elif stripped.startswith(_SHELL_PREFIXES):
            effects.append(f"line {number}: shell command: {stripped[:60]}")
    return effects, setup


def plan_cells(
    cells: list[str],
    transform: Callable[[str], str],
    *,
    name: str = "notebook",
    options: InstrumentOptions | None = None,
) -> ScriptPlan:
    """Plan notebook code cells (one :class:`StatementInfo` per cell).

    ``transform`` turns IPython syntax (magics, ``!cmd``) into Python, as
    ``InteractiveShell.transform_cell`` does. Loop ids use :func:`cell_file_id` and
    ``ast.unparse``-based headers, matching :class:`CellInstrumenter` at run time.
    """
    statements: list[StatementInfo] = []
    functions: dict[str, FunctionInfo] = {}
    imported: set[str] = set()
    aliases: set[str] = set()
    names: set[str] = set()
    trees: list[ast.Module] = []
    for raw in cells:
        tree = ast.parse(transform(raw))
        trees.append(tree)
        cell_aliases, cell_names = _sdk_aliases(tree)
        aliases |= cell_aliases
        names |= cell_names
    for index, (raw, tree) in enumerate(zip(cells, trees, strict=True)):
        scan = _Scan()
        scan.visit(tree)
        imported |= scan.imports
        inst = _Instrumenter(
            None, cell_file_id(index), frozenset(aliases), frozenset(names), options
        )
        inst.block(tree.body)
        for fname, info in _function_infos(tree, None, cell_file_id(index)).items():
            functions[fname] = info
        effects, setup = _cell_commands(raw)
        body = tree.body
        defs: list[ast.stmt] = [node for node in body if isinstance(node, _DEFINITIONS)]
        only_defs = bool(body) and len(defs) == len(body)
        definitions_code = None
        if defs and not only_defs:
            module = ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[]))
            definitions_code = compile(module, f"<{cell_file_id(index)} definitions>", "exec")
        statements.append(
            StatementInfo(
                index=index,
                lineno=body[0].lineno if body else 1,
                end_lineno=(body[-1].end_lineno or 1) if body else 1,
                kind="Cell",
                code=None,
                assigned=frozenset(scan.assigned),
                call_args=frozenset(scan.call_args),
                calls=frozenset(scan.calls),
                method_calls=frozenset(scan.method_calls),
                loops=inst.loops,
                uninstrumented=inst.uninstrumented,
                side_effects=scan.side_effects + effects,
                is_definition=only_defs,
                is_setup=setup,
                source=raw,
                definitions_code=definitions_code,
            )
        )
    return ScriptPlan(
        filename=name,
        statements=statements,
        functions=functions,
        global_names=frozenset(n for f in functions.values() for n in f.global_names),
        imported_modules=frozenset(imported),
        sdk_aliases=frozenset(aliases),
        sdk_names=frozenset(names),
        options=options or InstrumentOptions(),
    )


class CellInstrumenter(ast.NodeTransformer):
    """IPython AST transformer: instruments the loops of the current cell."""

    def __init__(self, plan: ScriptPlan) -> None:
        self.plan = plan
        self.cell: int | None = None

    def visit_Module(self, node: ast.Module) -> ast.Module:
        if self.cell is None:
            return node
        inst = _Instrumenter(
            None,
            cell_file_id(self.cell),
            self.plan.sdk_aliases,
            self.plan.sdk_names,
            self.plan.options,
        )
        inst.block(node.body)
        return ast.fix_missing_locations(node)


def called_functions(stmt: StatementInfo, plan: ScriptPlan) -> list[FunctionInfo]:
    """Module-level functions reachable from ``stmt`` through direct calls."""
    seen: dict[str, FunctionInfo] = {}
    todo = [name for name in stmt.calls if name in plan.functions]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        info = plan.functions[name]
        seen[name] = info
        todo.extend(c for c in info.calls if c in plan.functions and c not in seen)
    return list(seen.values())
