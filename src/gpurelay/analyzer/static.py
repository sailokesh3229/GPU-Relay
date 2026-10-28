"""Static resumability analysis of a script (spec §14).

Levels per top-level block (statement):

* ``FULL`` — safe points inside the block (instrumented ``for`` loop = layer C, framework
  training call = layer B, SDK loop = layer D), or a simple statement (checkpoints happen at
  statement boundaries).
* ``PARTIAL`` — long work without safe points inside (``while`` loops, ``async for``, loops in
  called functions): after an interruption the block restarts from its beginning.

Overall: the weakest block, except ``RESTART_ONLY`` when the script has no safe points inside
any block and all of its looping work sits in a single block (e.g. ``main()``): an
interruption then restarts the whole job.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gpurelay.models import ResumabilityLevel
from gpurelay.runtime.instrument import (
    FRAMEWORK_MODULES,
    FunctionInfo,
    InstrumentOptions,
    ScriptPlan,
    StatementInfo,
    called_functions,
    plan_cells,
    plan_script,
)

ANALYSIS_FORMAT_VERSION = 1

# Framework training entry points covered by Layer B hooks: (method name, framework).
_FRAMEWORK_CALLS = {("train", "hf"), ("fit", "lightning"), ("fit", "keras")}
_FRAMEWORK_DISTS = {
    "hf": ("transformers",),
    "lightning": ("lightning", "pytorch-lightning"),
    "keras": ("keras",),
}


def _supported_majors() -> dict[str, tuple[int, ...]]:
    from gpurelay.runtime.hooks import hf, keras, lightning

    return {
        "hf": hf.SUPPORTED_MAJOR,
        "lightning": lightning.SUPPORTED_MAJOR,
        "keras": keras.SUPPORTED_MAJOR,
    }


@dataclass
class Finding:
    kind: str
    message: str
    line: int | None = None
    cell: int | None = None  # 1-based code-cell number (notebooks)


@dataclass
class BlockReport:
    index: int
    lines: tuple[int, int]
    statement: str
    level: ResumabilityLevel
    cell: int | None = None  # 1-based code-cell number (notebooks)
    covered_by: list[str] = field(default_factory=list)
    loops: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    path: str
    kind: str
    overall: ResumabilityLevel
    blocks: list[BlockReport]
    simple_statements: int
    frameworks: dict[str, dict[str, Any]]
    risks: list[Finding]
    suggestions: list[Finding]
    environment: dict[str, Any]
    # Problems that stop the job as written (analyzer/portability.py), shown before risks.
    errors: list[Finding] = field(default_factory=list)
    format: int = ANALYSIS_FORMAT_VERSION

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["overall"] = self.overall.value
        for block in data["blocks"]:
            block["level"] = block["level"].value
            block["lines"] = list(block["lines"])
        return data


# ====================================================================== blocks


def _framework_calls(stmt: StatementInfo, frameworks: set[str]) -> list[str]:
    return sorted(
        fw for method, fw in _FRAMEWORK_CALLS if fw in frameworks and method in stmt.method_calls
    )


def _block(stmt: StatementInfo, plan: ScriptPlan, frameworks: set[str]) -> BlockReport | None:
    funcs_with_loops = [f for f in called_functions(stmt, plan) if f.loops]
    opts = plan.options

    def covered(fn: FunctionInfo) -> bool:  # loops of fn get safe points (opt-in options)
        kinds = {"for"} | ({"while"} if opts.while_loops else set())
        return opts.functions and fn.instrumentable and all(lp.kind in kinds for lp in fn.loops)

    covered_funcs = [f for f in funcs_with_loops if covered(f)]
    funcs_with_loops = [f for f in funcs_with_loops if not covered(f)]
    fw_calls = _framework_calls(stmt, frameworks)
    interesting = (
        stmt.loops
        or stmt.uninstrumented
        or funcs_with_loops
        or covered_funcs
        or fw_calls
        or stmt.side_effects
        or stmt.is_setup
    )
    if not interesting:
        return None
    report = BlockReport(
        index=stmt.index,
        lines=(stmt.lineno, stmt.end_lineno),
        statement=stmt.kind,
        level=ResumabilityLevel.FULL,
        cell=stmt.index + 1 if stmt.kind == "Cell" else None,
    )
    for loop in stmt.loops:
        layer = "D (SDK)" if loop.kind == "sdk" else "C (loop instrumentation)"
        if layer not in report.covered_by:
            report.covered_by.append(layer)
        report.loops.append(
            {"line": loop.lineno, "header": loop.header, "kind": loop.kind, "safe_points": True}
        )
    for fn in covered_funcs:
        if "C (loops in functions)" not in report.covered_by:
            report.covered_by.append("C (loops in functions)")
        for loop in fn.loops:
            report.loops.append({"line": loop.lineno, "header": loop.header, "kind": loop.kind,
                                 "safe_points": True, "function": fn.name})  # fmt: skip
    for fw in fw_calls:
        report.covered_by.append(f"B ({fw} hook)")
    for loop in stmt.uninstrumented:
        report.loops.append(
            {"line": loop.lineno, "header": loop.header, "kind": loop.kind, "safe_points": False}
        )
        hint = (
            " (set `instrument_while = true` under [runtime] to give it safe points)"
            if loop.kind == "while" and not opts.while_loops
            else ""
        )
        report.notes.append(
            f"line {loop.lineno}: this `{loop.kind}` loop has no safe points{hint}; "
            "the block restarts from its beginning after an interruption"
        )
    if funcs_with_loops and not fw_calls:
        names = ", ".join(sorted(f"{f.name}()" for f in funcs_with_loops))
        hint = "set `instrument_functions = true` under [runtime], " if not opts.functions else ""
        report.notes.append(
            f"calls {names} containing loops without safe points ({hint}use "
            "relay.range/relay.chunks inside them, or move the loop to the top level)"
        )
    if stmt.uninstrumented or (funcs_with_loops and not fw_calls):
        report.level = ResumabilityLevel.PARTIAL
    for effect in stmt.side_effects:
        report.notes.append(f"{effect} (never replayed automatically)")
    if stmt.is_setup:
        report.notes.append("installs packages: re-run in every session (environment setup)")
    return report


def _overall(blocks: list[BlockReport]) -> ResumabilityLevel:
    partial = [b for b in blocks if b.level is ResumabilityLevel.PARTIAL]
    with_safe_points = [b for b in blocks if b.covered_by]
    if not partial:
        return ResumabilityLevel.FULL
    if not with_safe_points and len(partial) == 1:
        return ResumabilityLevel.RESTART_ONLY
    return ResumabilityLevel.PARTIAL


# ====================================================================== frameworks


def _framework_status(frameworks: set[str]) -> dict[str, dict[str, Any]]:
    majors = _supported_majors()
    status: dict[str, dict[str, Any]] = {}
    for fw in sorted(frameworks):
        installed = None
        for dist in _FRAMEWORK_DISTS[fw]:
            try:
                installed = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        supported = None
        if installed is not None:
            try:
                supported = int(installed.split(".")[0]) in majors[fw]
            except ValueError:
                supported = False
        status[fw] = {
            "installed_here": installed,
            "supported_majors": list(majors[fw]),
            "hook_supported": supported,
        }
    return status


# ====================================================================== risks & suggestions


_SEED_CALLS = {"seed", "manual_seed", "seed_everything", "set_seed", "set_random_seed"}
_RNG_HINTS = re.compile(
    r"\b(random\.(random|randint|choice|shuffle|uniform)|np\.random\.|torch\.(rand|randn|randint|"
    r"randperm|bernoulli|normal)|Dropout\(|shuffle\s*=\s*True)"
)


def _risks(tree: ast.Module, text: str, plan: ScriptPlan) -> list[Finding]:
    risks = _tree_risks(tree, text)
    risks += _seed_risk([tree], text)
    risks += _plan_risks(plan)
    return risks


def _tree_risks(tree: ast.Module, text: str, cell: int | None = None) -> list[Finding]:
    risks: list[Finding] = []
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
                target = sub.targets[0]
                value = sub.value
                name = target.id if isinstance(target, ast.Name) else None
                if name is None:
                    continue
                unpicklable = None
                if isinstance(value, ast.GeneratorExp):
                    unpicklable = "generator"
                elif (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in ("iter", "open")
                ):
                    unpicklable = f"{value.func.id}() result"
                elif (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr in ("Lock", "RLock", "Thread", "Event", "Popen")
                ):
                    unpicklable = f"{value.func.attr} object"
                if unpicklable:
                    risks.append(
                        Finding(
                            "unpicklable_state",
                            f"`{name}` is a {unpicklable}; it cannot be checkpointed and will "
                            "be recreated (or missing) after a resume",
                            sub.lineno,
                            cell,
                        )
                    )
            if (
                isinstance(sub, ast.Assign)
                and any(
                    (ast.get_source_segment(text, t) or "").endswith("cudnn.benchmark")
                    for t in sub.targets
                )
                and isinstance(sub.value, ast.Constant)
                and sub.value.value is True
            ):
                risks.append(
                    Finding(
                        "nondeterminism",
                        "torch.backends.cudnn.benchmark = True selects kernels at runtime; "
                        "results can differ after a resume (set it to False for exact equivalence)",
                        sub.lineno,
                        cell,
                    )
                )
    return risks


def _seed_risk(trees: list[ast.Module], text: str) -> list[Finding]:
    seeded = any(
        isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Attribute) and n.func.attr in _SEED_CALLS)
             or (isinstance(n.func, ast.Name) and n.func.id in _SEED_CALLS))
        for tree in trees
        for n in ast.walk(tree)
    )  # fmt: skip
    if not seeded and _RNG_HINTS.search(text):
        return [
            Finding(
                "nondeterminism",
                "randomness is used but no RNG is seeded; resumed runs restore RNG state, but "
                "an uninterrupted reference run is not reproducible (seed random/numpy/torch)",
            )
        ]
    return []


def _plan_risks(plan: ScriptPlan) -> list[Finding]:
    risks: list[Finding] = []
    for stmt in plan.statements:
        cell = stmt.index + 1 if stmt.kind == "Cell" else None
        for effect in stmt.side_effects:
            risks.append(Finding("side_effect", effect, cell=cell))
        for loop in stmt.uninstrumented:
            risks.append(
                Finding(
                    "uninstrumented_loop",
                    f"`{loop.header}` is not instrumented in v1 (restarts as a block)",
                    loop.lineno,
                    cell,
                )
            )
    return risks


def _suggestions(tree: ast.Module, text: str, plan: ScriptPlan) -> list[Finding]:
    return _text_suggestions(text, plan) + _tree_suggestions(tree)


def _text_suggestions(text: str, plan: ScriptPlan) -> list[Finding]:
    out: list[Finding] = []
    uses_torch = "torch" in plan.imported_modules
    if uses_torch and ".backward(" in text and not re.search(r"autocast|GradScaler|\bamp\b", text):
        out.append(
            Finding(
                "mixed_precision",
                "training without mixed precision: torch.autocast + GradScaler usually cuts "
                "GPU time substantially on GPUs with tensor cores (e.g. T4)",
            )
        )
    if re.search(r"inception|\bfid\b|frechet", text, re.IGNORECASE):
        out.append(
            Finding(
                "cache_features",
                "FID-style metric: real-image features never change; compute them once, save "
                "them (e.g. np.save), and reuse them across runs instead of recomputing",
            )
        )
    return out


def _tree_suggestions(tree: ast.Module, cell: int | None = None) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        )
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if name == "DataLoader":
            workers = keywords.get("num_workers")
            if workers is None or (isinstance(workers, ast.Constant) and workers.value == 0):
                out.append(
                    Finding(
                        "dataloader_workers",
                        "DataLoader without worker processes: num_workers=2-4 keeps the GPU "
                        "fed (Kaggle/Colab sessions have 2-4 CPU cores)",
                        node.lineno,
                        cell,
                    )
                )
            pin = keywords.get("pin_memory")
            if not (isinstance(pin, ast.Constant) and pin.value is True):
                out.append(
                    Finding(
                        "pin_memory",
                        "DataLoader(pin_memory=True) speeds up host-to-GPU copies",
                        node.lineno,
                        cell,
                    )
                )
        eval_steps = keywords.get("eval_steps")
        if (
            isinstance(eval_steps, ast.Constant)
            and isinstance(eval_steps.value, int | float)
            and 0 < eval_steps.value <= 10
        ):
            out.append(
                Finding(
                    "frequent_evaluation",
                    f"evaluation every {eval_steps.value} step(s) costs GPU time; evaluate "
                    "less often",
                    node.lineno,
                    cell,
                )
            )
    return out


# ====================================================================== environment

_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "gpurelay": "gpurelay",
}


def _normalize(dist: str) -> str:
    return re.sub(r"[-_.]+", "-", dist).lower()


def _environment(plan: ScriptPlan, script: Path, requirements: Path | None) -> dict[str, Any]:
    stdlib: frozenset[str] = getattr(sys, "stdlib_module_names", frozenset())
    local = {p.stem for p in script.parent.glob("*.py")} | {
        p.name for p in script.parent.iterdir() if p.is_dir()
    }
    third_party = sorted(
        m for m in plan.imported_modules if m not in stdlib and m not in local and m != "__future__"
    )
    env: dict[str, Any] = {"third_party_imports": third_party}
    if requirements is not None:
        listed = set()
        for raw in requirements.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            listed.add(_normalize(re.split(r"[\s\[<>=!~;@]", line, maxsplit=1)[0]))
        missing = [m for m in third_party if _normalize(_IMPORT_TO_DIST.get(m, m)) not in listed]
        env["requirements"] = str(requirements.name)
        env["missing_from_requirements"] = missing
    return env


# ====================================================================== entry point


def _will_fail(trees: list[ast.Module], *, cells: bool) -> list[Finding]:
    from gpurelay.analyzer.portability import will_fail

    return will_fail(trees, cells=cells)


def analyze_script(
    path: Path, *, requirements: Path | None = None, options: InstrumentOptions | None = None
) -> Analysis:
    """Analyze a ``.py`` script."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    plan = plan_script(text, str(path), loop_file_id=path.name, options=options)
    tree = ast.parse(text)
    frameworks = plan.frameworks()
    blocks = [b for stmt in plan.statements if (b := _block(stmt, plan, frameworks))]
    simple = sum(1 for s in plan.statements if not s.is_definition) - len(blocks)
    return Analysis(
        path=path.name,
        kind="script",
        overall=_overall(blocks),
        blocks=blocks,
        simple_statements=simple,
        frameworks=_framework_status(frameworks),
        risks=_risks(tree, text, plan),
        suggestions=_suggestions(tree, text, plan),
        environment=_environment(plan, path, requirements),
        errors=_will_fail([tree], cells=False),
    )


def analyze_notebook(
    path: Path, *, requirements: Path | None = None, options: InstrumentOptions | None = None
) -> Analysis:
    """Analyze a ``.ipynb`` notebook (one block per code cell)."""
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    path = Path(path)
    notebook = nbformat.read(path, as_version=4)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    transform = TransformerManager().transform_cell
    plan = plan_cells(sources, transform, name=path.name, options=options)
    texts = [transform(src) for src in sources]
    trees = [ast.parse(text) for text in texts]
    frameworks = plan.frameworks()
    blocks = [b for stmt in plan.statements if (b := _block(stmt, plan, frameworks))]
    simple = sum(1 for s in plan.statements if not s.is_definition) - len(blocks)
    joined = "\n".join(texts)
    risks: list[Finding] = []
    suggestions = _text_suggestions(joined, plan)
    for number, (tree, text) in enumerate(zip(trees, texts, strict=True), start=1):
        risks += _tree_risks(tree, text, number)
        suggestions += _tree_suggestions(tree, number)
    risks += _seed_risk(trees, joined) + _plan_risks(plan)
    return Analysis(
        path=path.name,
        kind="notebook",
        overall=_overall(blocks),
        blocks=blocks,
        simple_statements=simple,
        frameworks=_framework_status(frameworks),
        risks=risks,
        suggestions=suggestions,
        environment=_environment(plan, path, requirements),
        errors=_will_fail(trees, cells=True),
    )


def analyze_path(
    path: Path, *, requirements: Path | None = None, options: InstrumentOptions | None = None
) -> Analysis:
    """Analyze a script or notebook (by extension), with the runtime's instrumentation
    ``options`` (default: the defaults of ``[runtime]``)."""
    if Path(path).suffix == ".ipynb":
        return analyze_notebook(path, requirements=requirements, options=options)
    return analyze_script(path, requirements=requirements, options=options)


__all__ = [
    "FRAMEWORK_MODULES",
    "Analysis",
    "BlockReport",
    "Finding",
    "analyze_notebook",
    "analyze_path",
    "analyze_script",
]
