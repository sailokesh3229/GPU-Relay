"""Command-line interface (spec §15).

Every command from the spec is registered now; commands scheduled for a later
implementation phase print ``not yet implemented (phase N)`` and exit with code 1.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape

from gpurelay import __version__

app = typer.Typer(
    name="gpurelay",
    help="Run ML workloads to completion on free GPU tiers by relaying them across platforms.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"gpurelay {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """GPU Relay: checkpoint automatically, hand off before limits, resume elsewhere."""


# ---------------------------------------------------------------- setup


@app.command()
def init(
    hf_repo: Annotated[
        str | None,
        typer.Option("--hf-repo", help="Storage repo name or id (default <user>/gpurelay-store)."),
    ] = None,
    kaggle: Annotated[
        bool | None,
        typer.Option("--kaggle/--no-kaggle", help="Set up the Kaggle backend (default: if "
                     "Kaggle credentials are found)."),
    ] = None,
    rotate_kaggle_secret: Annotated[
        bool,
        typer.Option("--rotate-kaggle-secret",
                     help="Replace the token stored in the Kaggle secrets dataset."),
    ] = False,
    local: Annotated[
        bool,
        typer.Option("--local", help="Storage on this computer only: no accounts, runs jobs "
                     "here (try GPU Relay out, or use your own GPU)."),
    ] = False,
    gdrive: Annotated[
        Path | None,
        typer.Option("--gdrive", help="Use Google Drive storage: path to an OAuth client "
                     "secret JSON (\"Desktop app\") from Google Cloud Console."),
    ] = None,
) -> None:  # fmt: skip
    """Create config, set up the storage repo, check backend credentials."""
    from gpurelay.config import gpurelay_home, load_config, save_config

    if local:
        config = load_config()
        config.storage.kind = "localfs"
        config.backends.local.enabled = True
        path = save_config(config)
        store = config.storage.localfs_root or gpurelay_home() / "store"
        console.print(f"[green]local storage: {store}[/green]")
        console.print(f"[green]config written to {path}[/green]")
        console.print(
            "next: `gpurelay submit your_script.py`, then `gpurelay up --until-idle` "
            "(jobs run on this computer; `gpurelay init` later adds Kaggle/Colab)"
        )
        return
    if gdrive is not None:
        import json

        from gpurelay.controller.setup import setup_gdrive
        from gpurelay.credentials import GDRIVE_CREDENTIALS, set_secret
        from gpurelay.errors import GpuRelayError

        try:
            from gpurelay.storage.gdrive import OfficialDriveApi, authorize, credentials_from_info
        except ImportError as exc:
            _fail(RuntimeError(f"Google Drive needs extra packages ({exc.name}): "
                               "pip install 'gpurelay[gdrive]'"))  # fmt: skip
        config = load_config()
        console.print("A browser window opens: sign in and allow GPU Relay to manage the files "
                      "it creates in your Drive.")  # fmt: skip
        try:
            info = authorize(gdrive)
            folder_id = setup_gdrive(config, OfficialDriveApi(credentials_from_info(info)))
        except GpuRelayError as exc:
            _fail(exc)
        except Exception as exc:
            _fail(RuntimeError(f"Google Drive setup failed: {exc}"))
        if not set_secret(GDRIVE_CREDENTIALS, json.dumps(info)):
            console.print("[yellow]no OS keyring: export GPURELAY_GDRIVE_CREDENTIALS with the "
                          "credentials JSON before using GPU Relay[/yellow]")  # fmt: skip
        path = save_config(config)
        console.print(f"[green]storage: Google Drive folder gpurelay-store ({folder_id})[/green]")
        console.print(f"[green]config written to {path}[/green]")
        console.print("[dim]Kaggle, Colab and Lightning sessions need Hugging Face storage "
                      "(`gpurelay init`); Drive serves local backends.[/dim]")  # fmt: skip
        return

    from gpurelay.credentials import (
        HF_TOKEN,
        hf_token,
        kaggle_credentials_present,
        set_secret,
    )
    from gpurelay.errors import GpuRelayError

    config = load_config()
    # 1. Hugging Face storage.
    token = hf_token()
    if token:
        console.print("[dim]using the Hugging Face token from HF_TOKEN / the keyring[/dim]")
    else:
        console.print(
            "Create a Hugging Face token with *write* access at "
            "https://huggingface.co/settings/tokens"
        )
        token = typer.prompt("Hugging Face token", hide_input=True).strip()
    try:
        from huggingface_hub import HfApi

        from gpurelay.controller.setup import (
            setup_kaggle,
            setup_storage,
            validate_session_token,
        )

        storage = setup_storage(config, HfApi(token=token), repo=hf_repo)
    except GpuRelayError as exc:
        _fail(exc)
    except Exception as exc:
        _fail(RuntimeError(f"Hugging Face setup failed: {exc}"))
    verb = "created" if storage.created else "verified"
    console.print(f"[green]storage: private dataset repo {storage.repo_id} ({verb})[/green]")
    if set_secret(HF_TOKEN, token):
        console.print("[green]token stored in the OS keyring[/green]")
    else:
        console.print(
            "[yellow]no OS keyring available: export HF_TOKEN before using GPU Relay[/yellow]"
        )

    # 2. Kaggle backend + secrets dataset.
    use_kaggle = kaggle if kaggle is not None else kaggle_credentials_present()
    if kaggle is None and not use_kaggle:
        console.print(
            "[yellow]kaggle: no credentials found, skipping. Create a token at "
            "https://www.kaggle.com/settings/api, save it to ~/.kaggle/access_token, and run "
            "`gpurelay init` again.[/yellow]"
        )
    if use_kaggle and not kaggle_credentials_present():
        console.print(
            "[yellow]no Kaggle credentials: create a token at https://www.kaggle.com/settings/api "
            "and save it to ~/.kaggle/access_token, then run `gpurelay init` again[/yellow]"
        )
        use_kaggle = False
    if use_kaggle:
        from gpurelay.backends.kaggle import OfficialKaggleClient

        console.print(
            "Kaggle sessions need a [bold]Hugging Face[/bold] token (starts with hf_) that can "
            f"write to {storage.repo_id}. Recommended: a fine-grained token limited to that repo "
            "(https://huggingface.co/settings/tokens). This is NOT your Kaggle token."
        )
        while True:
            session_token = (
                typer.prompt(
                    "Hugging Face token for Kaggle sessions (Enter = reuse the token above)",
                    default="",
                    hide_input=True,
                    show_default=False,
                ).strip()
                or token
            )
            try:
                validate_session_token(session_token, storage.repo_id, HfApi())
                break
            except GpuRelayError as exc:
                console.print(f"[red]{escape(str(exc))}[/red] - try again")
        try:
            result = setup_kaggle(
                config, OfficialKaggleClient(), session_token=session_token,
                repo_id=storage.repo_id, rotate=rotate_kaggle_secret,
            )  # fmt: skip
        except GpuRelayError as exc:
            _fail(exc)
        verb = (
            ("updated" if rotate_kaggle_secret else "created")
            if result.created
            else "already exists; --rotate-kaggle-secret replaces its token"
        )
        console.print(
            f"[green]kaggle: {result.username}; private secrets dataset "
            f"{result.secrets_dataset} ({verb})[/green]"
        )
    path = save_config(config)
    console.print(f"[green]config written to {path}[/green]")
    console.print(
        "[dim]Kaggle limits in the config are estimates (see docs/verify.md); "
        "next: docs/manual-tests.md[/dim]"
    )


@app.command()
def analyze(
    path: Annotated[Path, typer.Argument(help="Script (.py) or notebook (.ipynb) to analyze.")],
    probe: Annotated[
        bool,
        typer.Option("--probe", help="Also run the job here briefly to measure speed and memory."),
    ] = False,
    probe_time: Annotated[
        str, typer.Option("--probe-time", help="How long the --probe run lasts (e.g. 60s, 5m).")
    ] = "60s",
    requirements: Annotated[
        Path | None,
        typer.Option("--requirements", help="requirements.txt to check imports against."),
    ] = None,
    json_out: Annotated[
        Path, typer.Option("--json", help="Where to write the machine-readable report.")
    ] = Path("analysis.json"),
    no_json: Annotated[bool, typer.Option("--no-json", help="Do not write analysis.json.")] = False,
) -> None:
    """Report resumability level, risks, and GPU-time savings suggestions."""
    import json

    from gpurelay.analyzer import analyze_path
    from gpurelay.analyzer.report import render

    if not path.is_file():
        raise typer.BadParameter(f"{path} does not exist", param_hint="PATH")
    from gpurelay.config import load_config
    from gpurelay.errors import GpuRelayError
    from gpurelay.runtime.instrument import InstrumentOptions

    try:
        options = InstrumentOptions.from_settings(load_config().runtime)
    except GpuRelayError:
        options = InstrumentOptions()  # unreadable config: analyze with the defaults
    try:
        analysis = analyze_path(path, requirements=requirements, options=options)
    except SyntaxError as exc:
        console.print(f"[red]{escape(f'{path}: syntax error: {exc}')}[/red]")
        raise typer.Exit(code=1) from exc
    render(analysis, Console())
    report = analysis.to_json()
    if probe:
        from gpurelay.analyzer.probe import probe as run_probe
        from gpurelay.analyzer.report import render_probe
        from gpurelay.durations import parse_duration

        try:
            seconds = parse_duration(probe_time)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--probe-time") from exc
        console.print(f"[cyan]probe: running {path.name} here for ~{probe_time}...[/cyan]")
        result = run_probe(path, seconds=seconds, requirements=requirements)
        render_probe(result, Console())
        report["probe"] = result.to_json()
    if not no_json:
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"[dim]wrote {json_out}[/dim]")


# ---------------------------------------------------------------- jobs


def _context() -> Any:
    """Config + state DB + storage + controller (lazy imports keep `--help` fast)."""
    from gpurelay.config import load_config
    from gpurelay.controller.service import open_context
    from gpurelay.errors import GpuRelayError

    try:
        return open_context(load_config())
    except GpuRelayError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc


def _fail(exc: Exception) -> NoReturn:
    console.print(f"[red]{escape(str(exc))}[/red]")
    raise typer.Exit(code=1) from exc


@app.command()
def submit(
    path: Annotated[Path, typer.Argument(help="Script (.py) or notebook (.ipynb) to run.")],
    name: Annotated[str | None, typer.Option("--name", help="Job name.")] = None,
    priority: Annotated[int, typer.Option("--priority", help="Higher runs first.")] = 0,
    requirements: Annotated[
        Path | None, typer.Option("--requirements", help="requirements.txt for the job.")
    ] = None,
    data: Annotated[
        list[str] | None,
        typer.Option("--data", help="Data source: kaggle:|hf:|url:|local:<ref>. Repeatable."),
    ] = None,
    estimate: Annotated[
        str | None, typer.Option("--estimate", help="Expected GPU time, e.g. 20h.")
    ] = None,
    prefer: Annotated[str | None, typer.Option("--prefer", help="Preferred backend.")] = None,
    accept_restart_only: Annotated[
        bool,
        typer.Option("--accept-restart-only", help="Allow blocks that can only restart."),
    ] = False,
    include: Annotated[
        list[Path] | None,
        typer.Option("--include", help="Extra file to bundle (next to the entry). Repeatable."),
    ] = None,
) -> None:
    """Bundle a job, upload it to storage, and queue it."""
    import time

    from gpurelay.analyzer import analyze_path
    from gpurelay.controller.estimator import fit_message
    from gpurelay.controller.service import submit_job
    from gpurelay.controller.views import budget_lines
    from gpurelay.durations import parse_duration
    from gpurelay.errors import GpuRelayError
    from gpurelay.models import DataSpec, ResumabilityLevel

    if not path.is_file():
        raise typer.BadParameter(f"{path} does not exist", param_hint="PATH")
    try:
        specs = [DataSpec.parse(spec) for spec in data or []]
        estimate_h = parse_duration(estimate) / 3600 if estimate else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    analysis = analyze_path(path, requirements=requirements)
    if analysis.overall is ResumabilityLevel.RESTART_ONLY and not accept_restart_only:
        console.print(
            "[red]this job has no safe points: any interruption restarts it from scratch. "
            "See `gpurelay analyze`; pass --accept-restart-only to submit anyway.[/red]"
        )
        raise typer.Exit(code=1)
    ctx = _context()
    extra = {p.name: p for p in include or []}
    try:
        job = submit_job(
            ctx,
            path,
            name=name,
            priority=priority,
            requirements=requirements,
            data=specs,
            estimate_h=estimate_h,
            prefer=prefer,
            accept_restart_only=accept_restart_only,
            extra_files=extra,
        )
    except GpuRelayError as exc:
        _fail(exc)
    console.print(f"[green]submitted {job.id}[/green] ({analysis.overall.value})")
    console.print(fit_message(estimate_h, budget_lines(ctx.controller, time.time())))
    console.print("[dim]start the controller with `gpurelay up` if it is not running[/dim]")


@app.command()
def up(
    once: Annotated[bool, typer.Option("--once", help="Run a single scheduling tick.")] = False,
    until_idle: Annotated[
        bool, typer.Option("--until-idle", help="Exit when no job is queued or running.")
    ] = False,
    interval: Annotated[
        float | None, typer.Option("--interval", help="Seconds between ticks.")
    ] = None,
    takeover: Annotated[
        bool,
        typer.Option(
            "--takeover",
            help="Take the store's controller lease without waiting "
            "(the other controller, e.g. on a VM, stops at its next renewal).",
        ),
    ] = False,
) -> None:
    """Start the controller daemon (foreground)."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [gpurelay] %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "hf_xet", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from gpurelay.controller.lease import ControllerLease, LeaseLostError
    from gpurelay.controller.lock import controller_lock
    from gpurelay.errors import GpuRelayError

    ctx = _context()
    try:
        with controller_lock(ctx.home):
            lease = ControllerLease(ctx.storage)
            lease.acquire(takeover=takeover)  # one controller per store, across machines
            ctx.controller.lease = lease
            console.print(
                f"[cyan]controller: {len(ctx.controller.backends)} backend(s) "
                f"({', '.join(ctx.controller.backends) or 'none'})[/cyan]"
            )
            ctx.controller.run(
                interval_s=interval, until_idle=until_idle, max_ticks=1 if once else None
            )
    except LeaseLostError as exc:
        console.print(f"[yellow]{escape(str(exc))}; this controller stops[/yellow]")
    except GpuRelayError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        console.print("[yellow]controller stopped; running sessions continue[/yellow]")
    finally:
        lease_held = getattr(ctx.controller, "lease", None)
        if lease_held is not None:
            import contextlib

            with contextlib.suppress(Exception):  # best effort: a stale lease expires itself
                lease_held.release()


@app.command()
def status(
    job: Annotated[str | None, typer.Argument(help="Job id; all jobs if omitted.")] = None,
) -> None:
    """Show jobs, backends, progress, ETA, and remaining GPU budget."""
    import time

    from gpurelay.controller.service import resolve_job
    from gpurelay.controller.views import render_job, render_status
    from gpurelay.errors import GpuRelayError

    ctx = _context()
    out = Console()
    if job is None:
        render_status(out, ctx.controller, time.time())
        return
    try:
        render_job(out, ctx.controller, resolve_job(ctx, job), time.time())
    except GpuRelayError as exc:
        _fail(exc)


@app.command()
def logs(
    job: Annotated[str, typer.Argument(help="Job id.")],
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Follow the log.")] = False,
) -> None:
    """Show a job's merged log (committed output so far, or the final log)."""
    import sys
    import time

    from gpurelay.controller.service import merged_log, resolve_job
    from gpurelay.errors import GpuRelayError

    ctx = _context()
    try:
        target = resolve_job(ctx, job)
    except GpuRelayError as exc:
        _fail(exc)
    shown = 0
    while True:
        text = merged_log(ctx, target)
        if len(text) > shown:
            sys.stdout.write(text[shown:])
            sys.stdout.flush()
            shown = len(text)
        target = ctx.db.get_job(target.id) or target
        if not follow or target.state.is_terminal:
            return
        time.sleep(2)


class QuotaAction(str, Enum):
    show = "show"
    set = "set"


@app.command()
def quota(
    action: Annotated[QuotaAction, typer.Argument(help="show or set.")] = QuotaAction.show,
    backend: Annotated[str | None, typer.Argument(help="Backend name (for 'set').")] = None,
    remaining: Annotated[
        str | None, typer.Option("--remaining", help="Remaining budget, e.g. 12h.")
    ] = None,
    used: Annotated[str | None, typer.Option("--used", help="Used budget, e.g. 3h.")] = None,
) -> None:
    """Show GPU budget per backend, or correct it manually."""
    import time

    from gpurelay.controller.service import correct_quota
    from gpurelay.controller.views import render_status
    from gpurelay.durations import parse_duration
    from gpurelay.errors import GpuRelayError

    ctx = _context()
    if action is QuotaAction.set:
        if backend is None:
            raise typer.BadParameter("quota set needs a backend name")
        try:
            remaining_h = parse_duration(remaining, allow_zero=True) / 3600 if remaining else None
            used_h = parse_duration(used, allow_zero=True) / 3600 if used else None
            correct_quota(ctx, backend, remaining_h=remaining_h, used_h=used_h)
        except (ValueError, GpuRelayError) as exc:
            _fail(exc)
        console.print(f"[green]recorded correction for {backend}[/green]")
    render_status(Console(), ctx.controller, time.time())


def _request(job: str, action: str) -> None:
    from gpurelay.controller.service import request
    from gpurelay.errors import GpuRelayError

    ctx = _context()
    try:
        target = request(ctx, job, action)
    except GpuRelayError as exc:
        _fail(exc)
    console.print(f"[green]{action} requested for {target.id}[/green] (applied by the controller)")


@app.command()
def pause(job: Annotated[str, typer.Argument(help="Job id.")]) -> None:
    """Pause a job at its next checkpoint."""
    _request(job, "pause")


@app.command()
def resume(job: Annotated[str, typer.Argument(help="Job id.")]) -> None:
    """Resume a paused job."""
    _request(job, "resume")


@app.command()
def cancel(job: Annotated[str, typer.Argument(help="Job id.")]) -> None:
    """Cancel a job (checkpoints are kept)."""
    _request(job, "cancel")


@app.command()
def colab(job: Annotated[str, typer.Argument(help="Job id.")]) -> None:
    """Show the Colab resume notebook for a job, with instructions."""
    from gpurelay.controller.service import colab_session
    from gpurelay.errors import GpuRelayError

    ctx = _context()
    try:
        info = colab_session(ctx, job)
    except GpuRelayError as exc:
        _fail(exc)
    console.print(f"[bold]Resume notebook:[/bold] {info.notebook}")
    if info.waiting:
        console.print(f"[magenta]GPU Relay is waiting for you to run it (segment "
                      f"{info.segment_index}).[/magenta]")  # fmt: skip
    elif info.running:
        console.print(f"[green]running on Colab (segment {info.segment_index})[/green]")
    else:
        console.print(
            f"[yellow]No Colab session is requested right now (job: {info.job_state}). The "
            "notebook refuses to start until the controller asks; you get a notification."
            "[/yellow]"
        )
    steps = [
        "Open https://colab.research.google.com > File > Upload notebook, and choose the file "
        "above.",
        f"Once per Colab account: key icon (Secrets) > add [bold]{info.secret_name}[/bold] = a "
        f"Hugging Face token (hf_...) with write access to {info.repo_id}; turn on notebook "
        "access.",
        "Runtime > Change runtime type > a GPU.",
        "Runtime > Run all, and keep the tab open. Keep `gpurelay up` running here: it "
        "continues the job after the Colab session hands off.",
    ]
    console.print()
    for number, step in enumerate(steps, 1):
        console.print(f"{number}. {step}")


@app.command()
def fetch(
    job: Annotated[str, typer.Argument(help="Job id.")],
    to: Annotated[Path | None, typer.Option("--to", help="Destination directory.")] = None,
) -> None:
    """Download a job's final outputs."""
    from gpurelay.controller.service import fetch_outputs, resolve_job
    from gpurelay.errors import GpuRelayError

    ctx = _context()
    try:
        target = resolve_job(ctx, job)
        dest = to or Path(f"{target.name}-{target.id}")
        files = fetch_outputs(ctx, target, dest)
    except GpuRelayError as exc:
        _fail(exc)
    console.print(f"[green]{len(files)} file(s) -> {dest}[/green]")


# ---------------------------------------------------------------- testing / UI


@app.command("run-local")
def run_local(
    path: Annotated[Path, typer.Argument(help="Script (.py) or notebook (.ipynb) to run.")],
    simulate_kill_every: Annotated[
        str | None,
        typer.Option(
            "--simulate-kill-every", help="Hard-kill the session every interval, e.g. 5m."
        ),
    ] = None,
    simulate_deadline: Annotated[
        str | None,
        typer.Option("--simulate-deadline", help="Force a handoff deadline per segment, e.g. 10m."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Run directory (default: $GPURELAY_HOME/runs/<job>)."),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Do not stream output.")] = False,
    accept_restart_only: Annotated[
        bool,
        typer.Option("--accept-restart-only", help="Run even if the job can only restart."),
    ] = False,
) -> None:
    """Run a job locally, relaying segments until it completes (testing)."""
    import sys

    from gpurelay.backends.local import LocalBackend
    from gpurelay.backends.simulated import SimulatedBackend
    from gpurelay.config import gpurelay_home, load_config
    from gpurelay.controller.local_runner import SegmentRecord, run_local_job
    from gpurelay.durations import format_duration, parse_duration
    from gpurelay.models import JobState, new_id
    from gpurelay.storage import LocalFSStorage

    if not path.is_file():
        raise typer.BadParameter(f"{path} does not exist", param_hint="PATH")
    from gpurelay.analyzer import analyze_path
    from gpurelay.models import ResumabilityLevel

    analysis = analyze_path(path)
    console.print(f"[cyan][gpurelay] resumability: {analysis.overall.value}[/cyan]")
    if analysis.overall is ResumabilityLevel.RESTART_ONLY and not accept_restart_only:
        console.print(
            "[red]this job has no safe points: any interruption restarts it from scratch. "
            "See `gpurelay analyze`; pass --accept-restart-only to run anyway.[/red]"
        )
        raise typer.Exit(code=1)
    try:
        kill_s = parse_duration(simulate_kill_every) if simulate_kill_every else None
        deadline_s = parse_duration(simulate_deadline) if simulate_deadline else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    settings = load_config().runtime.model_copy()
    if kill_s is not None:
        # Commit often enough that every simulated session makes progress.
        settings.max_lost_work_s = min(settings.max_lost_work_s, kill_s / 4)
        settings.min_interval_s = min(settings.min_interval_s, settings.max_lost_work_s / 2)
    if deadline_s is not None:
        settings.safety_margin_s = min(settings.safety_margin_s, deadline_s / 5)
    if kill_s is not None or deadline_s is not None:
        console.print(
            "[dim]simulation settings: checkpoint every "
            f"{format_duration(settings.min_interval_s)}-{format_duration(settings.max_lost_work_s)}"
            f", safety margin {format_duration(settings.safety_margin_s)}[/dim]"
        )

    job_id = new_id("job")
    run_dir = (out or gpurelay_home() / "runs" / job_id).resolve()
    storage = LocalFSStorage(run_dir / "store")
    backend: LocalBackend
    if kill_s is not None or deadline_s is not None:
        backend = SimulatedBackend(
            storage,
            run_dir / "sessions",
            name="local-sim",
            settings=settings,
            max_session_s=deadline_s or 7 * 24 * 3600,
            kill_after_s=kill_s,
        )
    else:
        backend = LocalBackend(storage, run_dir / "sessions", settings=settings)

    def on_segment(record: SegmentRecord) -> None:
        console.print(
            f"[cyan][gpurelay] segment {record.segment_id}: {record.outcome} "
            f"after {format_duration(record.duration_s)} "
            f"(committed checkpoint: {record.committed_seq_after})[/cyan]"
        )

    def on_console(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    console.print(f"[cyan][gpurelay] job {job_id} -> {run_dir}[/cyan]")
    result = run_local_job(
        path,
        storage=storage,
        backend=backend,
        out_dir=run_dir / "final",
        job_id=job_id,
        on_segment=on_segment,
        on_console=None if quiet else on_console,
    )
    if result.state is JobState.COMPLETED:
        console.print(
            f"[green][gpurelay] completed in {len(result.segments)} segment(s); "
            f"outputs: {run_dir / 'final'}[/green]"
        )
        return
    console.print(f"[red][gpurelay] {result.state.value}: {result.message}[/red]")
    raise typer.Exit(code=1)


@app.command("export-state")
def export_state_cmd(
    archive: Annotated[Path, typer.Argument(help="Archive to write, e.g. gpurelay-state.tgz.")],
) -> None:
    """Pack the config and job database to move the controller (no secrets included)."""
    from gpurelay.config import gpurelay_home
    from gpurelay.controller.migrate import export_state
    from gpurelay.errors import GpuRelayError

    try:
        names = export_state(gpurelay_home(), archive)
    except GpuRelayError as exc:
        _fail(exc)
    console.print(f"[green]wrote {archive} ({', '.join(names)})[/green]")
    console.print(
        "On the other machine: `gpurelay import-state` it, set the tokens there (HF_TOKEN, "
        "~/.kaggle/access_token), stop the controller here, then start `gpurelay up` there."
    )


@app.command("import-state")
def import_state_cmd(
    archive: Annotated[Path, typer.Argument(help="Archive written by `gpurelay export-state`.")],
    force: Annotated[
        bool, typer.Option("--force", help="Replace a job database that already has jobs.")
    ] = False,
) -> None:
    """Install a config and job database exported on another machine."""
    from gpurelay.config import gpurelay_home
    from gpurelay.controller.migrate import import_state
    from gpurelay.errors import GpuRelayError

    try:
        names = import_state(gpurelay_home(), archive, force=force)
    except GpuRelayError as exc:
        _fail(exc)
    console.print(f"[green]imported {', '.join(names)} into {gpurelay_home()}[/green]")


@app.command()
def dashboard(
    port: Annotated[int, typer.Option("--port", help="Port on this computer.")] = 8765,
    host: Annotated[
        str, typer.Option("--host", help="Interface to listen on (no login: keep it local).")
    ] = "127.0.0.1",
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open it in your browser.")
    ] = True,
) -> None:
    """Open the local web dashboard (jobs, timeline, budgets, live log)."""
    try:
        import uvicorn

        from gpurelay.dashboard.app import create_app
    except ImportError as exc:
        _fail(RuntimeError(f"the dashboard needs extra packages ({exc.name}): "
                           "pip install 'gpurelay[dashboard]'"))  # fmt: skip
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[yellow]listening on {host}: anyone who can reach this address can see your jobs "
            "and pause or cancel them (the dashboard has no login)[/yellow]"
        )
    ctx = _context()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    console.print(f"[cyan]dashboard: {url}[/cyan]  (Ctrl+C to stop; `gpurelay up` runs the jobs)")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(ctx), host=host, port=port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    app()
