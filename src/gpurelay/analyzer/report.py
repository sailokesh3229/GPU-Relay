"""Human-readable rendering of an :class:`~gpurelay.analyzer.static.Analysis` (rich).

Everything that comes from the job (paths, its code, messages that quote it or config keys
like ``[runtime]``) is escaped: rich would read ``[...]`` as markup and drop it."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from gpurelay.analyzer.probe import ProbeResult
from gpurelay.analyzer.static import Analysis
from gpurelay.durations import format_duration
from gpurelay.models import ResumabilityLevel

_COLORS = {
    ResumabilityLevel.FULL: "green",
    ResumabilityLevel.PARTIAL: "yellow",
    ResumabilityLevel.RESTART_ONLY: "red",
}

_EXPLAIN = {
    ResumabilityLevel.FULL: "every block resumes at a safe point",
    ResumabilityLevel.PARTIAL: "some blocks restart from their beginning after an interruption",
    ResumabilityLevel.RESTART_ONLY: (
        "no safe points: an interruption restarts the whole job "
        "(submit with --accept-restart-only, or use relay.range/relay.chunks)"
    ),
}


def _where(cell: int | None, line: int | None) -> str:
    parts = ([f"cell {cell}"] if cell else []) + ([f"line {line}"] if line else [])
    return ", ".join(parts) + ": " if parts else ""


def render(analysis: Analysis, console: Console) -> None:
    color = _COLORS[analysis.overall]
    console.print(
        f"[bold]{escape(str(analysis.path))}[/bold]: "
        f"[{color} bold]{analysis.overall.value}[/{color} bold]"
        f" - {_EXPLAIN[analysis.overall]}"
    )
    if analysis.blocks:
        table = Table(show_header=True, header_style="bold", expand=False)
        table.add_column("cell" if analysis.kind == "notebook" else "lines", no_wrap=True)
        table.add_column("level", no_wrap=True)
        table.add_column("covered by")
        table.add_column("details")
        for block in analysis.blocks:
            c = _COLORS[block.level]
            details = [
                f"{'+' if loop['safe_points'] else '-'} line {loop['line']}: "
                f"{escape(str(loop['header']))}"
                for loop in block.loops
            ] + [escape(note) for note in block.notes]
            where = (
                f"cell {block.cell}"
                if block.cell is not None
                else f"{block.lines[0]}-{block.lines[1]}"
            )
            table.add_row(
                where,
                f"[{c}]{block.level.value}[/{c}]",
                escape(", ".join(block.covered_by)) or "-",
                "\n".join(details),
            )
        console.print(table)
    console.print(
        f"[dim]{analysis.simple_statements} other "
        f"{'cell' if analysis.kind == 'notebook' else 'statement'}(s): atomic, checkpointed "
        "at their boundaries[/dim]"
    )
    for fw, info in analysis.frameworks.items():
        installed = info["installed_here"] or "not installed here"
        verdict = {True: "hook supported", False: "hook NOT supported", None: "verify on backend"}[
            info["hook_supported"]
        ]
        console.print(f"framework [bold]{escape(fw)}[/bold]: {escape(str(installed))} ({verdict})")
    if analysis.errors:
        console.print("[bold red]Will fail as written[/bold red]")
        for error in analysis.errors:
            console.print(f"  [red]x[/red] {_where(error.cell, error.line)}{escape(error.message)}")
    if analysis.risks:
        console.print("[bold]Risks[/bold]")
        for risk in analysis.risks:
            where = _where(risk.cell, risk.line)
            console.print(f"  [yellow]![/yellow] {where}{escape(risk.message)}")
    if analysis.suggestions:
        console.print("[bold]GPU-time suggestions[/bold] (not applied automatically)")
        for tip in analysis.suggestions:
            where = _where(tip.cell, tip.line)
            console.print(f"  [cyan]*[/cyan] {where}{escape(tip.message)}")
    missing = analysis.environment.get("missing_from_requirements")
    if missing:
        console.print(
            f"[bold]Environment[/bold]: imports missing from "
            f"{escape(str(analysis.environment['requirements']))}: {escape(', '.join(missing))}"
        )


def render_probe(result: ProbeResult, console: Console) -> None:
    """The ``--probe`` measurements, with what they mean for a relay."""
    table = Table(title="Probe (this computer)", show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()

    def h(value: float | None) -> str:
        return "-" if value is None else format_duration(value * 3600)

    table.add_row("ran", f"{format_duration(result.seconds)}, {escape(result.ended)}")
    table.add_row("device", escape(result.device))
    startup = "-" if result.startup_s is None else format_duration(result.startup_s)
    table.add_row("startup", f"{startup} until the first loop progress")
    if result.completed is not None:
        done = f"{result.completed}/{result.total}" if result.total else str(result.completed)
        table.add_row("progress", f"{done} iterations of the outermost loop")
    rate = "-" if result.rate_per_h is None else f"{result.rate_per_h / 3600:.3g} iterations/s"
    table.add_row("rate", rate)
    table.add_row("projected total", h(result.projected_h))
    peak = "-" if result.peak_rss_mb is None else f"{result.peak_rss_mb:.0f} MB"
    table.add_row("peak memory (RAM)", peak)
    ckpt = "-" if result.checkpoint_mb is None else f"{result.checkpoint_mb:.1f} MB"
    table.add_row("checkpoint size", ckpt)
    console.print(table)
    if result.ended.startswith(("failed", "stopped")):
        console.print("[red]the probe run did not end cleanly; last output:[/red]")
        console.print(result.log_tail, markup=False, highlight=False)
    elif result.rate_per_h is None and result.ended != "completed":
        console.print(
            "[yellow]no loop progress was measured: the probe ended before the outermost "
            "loop completed an iteration, or the job has no instrumented loop[/yellow]"
        )
    if result.device == "CPU":
        console.print("[dim]measured on CPU: GPU sessions are usually much faster[/dim]")
