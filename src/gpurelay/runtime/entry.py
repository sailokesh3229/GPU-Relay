"""Runtime entry point inside a backend session (spec §12.1).

    python -m gpurelay.runtime.entry --job J --segment S --storage URI \
        --work-dir W --runtime-dir R [--deadline EPOCH_S] [--settings JSON]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from gpurelay import sdk
from gpurelay.bundle.packager import download_bundle, stage_data
from gpurelay.config import RuntimeSettings
from gpurelay.models import ExitCode, JobKind, RuntimeState
from gpurelay.retry import RetryPolicy
from gpurelay.runtime.checkpoint import load_latest_checkpoint
from gpurelay.runtime.env import env_fingerprint
from gpurelay.runtime.executor_script import run_script
from gpurelay.runtime.hooks import install_hooks
from gpurelay.runtime.parallel import ParallelRegion, inside_region, read_assignment
from gpurelay.runtime.session import RelaySession
from gpurelay.storage.base import Storage
from gpurelay.storage.factory import open_storage
from gpurelay.storage.layout import segment_env_key, segment_log_key, status_key

log = logging.getLogger("gpurelay.runtime")

# Session setup only reads inputs and writes the environment report, so each step can
# simply be repeated: one transient storage error must not cost the whole session.
SETUP_RETRY = RetryPolicy(attempts=5, base_delay_s=1.0)

_STATE_FOR_EXIT = {
    ExitCode.COMPLETED: RuntimeState.COMPLETED,
    ExitCode.HANDOFF: RuntimeState.HANDOFF,
    ExitCode.USER_ERROR: RuntimeState.FAILED_USER,
    ExitCode.NON_RESUMABLE: RuntimeState.FAILED_USER,
    ExitCode.INFRA_ERROR: RuntimeState.FAILED_INFRA,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="gpurelay.runtime.entry")
    p.add_argument("--job", required=True)
    p.add_argument("--segment", required=True)
    p.add_argument("--storage", required=True, help="storage URI, e.g. localfs:/path")
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--runtime-dir", required=True, type=Path)
    p.add_argument("--backend", default="local")
    p.add_argument("--deadline", type=float, default=None, help="session deadline (epoch s)")
    p.add_argument("--settings", default="{}", help="RuntimeSettings overrides as JSON")
    p.add_argument(
        "--install-requirements",
        action="store_true",
        help="pip-install the bundle's requirements (cloud sessions)",
    )
    return p.parse_args(argv)


def _setup_logging() -> None:
    # Runtime messages go to the real stderr, never into the captured user log.
    handler = logging.StreamHandler(sys.__stderr__)
    handler.setFormatter(logging.Formatter("[gpurelay] %(levelname)s %(message)s"))
    root = logging.getLogger("gpurelay")
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    root.propagate = False


def _write_completed_status(storage: Storage, job_id: str, segment_id: str) -> None:
    from gpurelay.models import StatusFile, utcnow

    status = StatusFile(
        job_id=job_id,
        segment_id=segment_id,
        state=RuntimeState.COMPLETED,
        heartbeat_at=utcnow(),
        message="already completed",
    )
    storage.put_bytes(status.model_dump_json().encode(), status_key(job_id))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging()
    storage = open_storage(args.storage)
    settings = RuntimeSettings.model_validate(json.loads(args.settings))
    work_dir: Path = args.work_dir.resolve()
    runtime_dir: Path = args.runtime_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle = SETUP_RETRY.call(
            lambda: download_bundle(storage, args.job, work_dir), what="bundle download"
        )
        staged = SETUP_RETRY.call(
            lambda: stage_data(storage, bundle, work_dir), what="data staging"
        )
        if args.install_requirements and bundle.requirements:
            from gpurelay.runtime.requirements import install_requirements

            install_requirements(work_dir / bundle.requirements)
        env = env_fingerprint() | {"backend": args.backend}
        SETUP_RETRY.call(
            lambda: storage.put_bytes(
                json.dumps(env, indent=2).encode(), segment_env_key(args.job, args.segment)
            ),
            what="environment upload",
        )
        loaded = load_latest_checkpoint(
            storage, args.job, runtime_dir / "restore", retry=SETUP_RETRY
        )
        worker_region = SETUP_RETRY.call(
            lambda: read_assignment(storage, args.job, args.segment), what="worker assignment"
        )
        if worker_region is not None:
            region = ParallelRegion(storage, args.job, worker_region, args.segment, 0)
            info = SETUP_RETRY.call(region.read_info, what="region info")
    except Exception as exc:
        log.error("session setup failed: %r", exc)
        return int(ExitCode.INFRA_ERROR)

    if worker_region is not None:
        # A parallel-chunks worker starts only from a checkpoint inside its region.
        position = loaded.positions.get("position") if loaded is not None else None
        if info is None or loaded is None or not inside_region(position, info):
            log.info("worker %s: region %s is not active; nothing to do", args.segment,
                     worker_region)  # fmt: skip
            return int(ExitCode.COMPLETED)
    elif loaded is not None and loaded.runtime.get("final"):
        log.info("job %s already completed (checkpoint %d)", args.job, loaded.seq)
        SETUP_RETRY.call(
            lambda: _write_completed_status(storage, args.job, args.segment),
            what="completed status",
        )
        return int(ExitCode.COMPLETED)

    session = RelaySession(
        storage=storage,
        job_id=args.job,
        segment_id=args.segment,
        work_dir=work_dir,
        runtime_dir=runtime_dir,
        settings=settings,
        deadline=args.deadline,
        loaded=loaded,
        baseline_files=bundle.files,
        exclude_files=set(staged),
        worker_region=worker_region,
    )
    if loaded is not None:
        log.info("resuming from checkpoint %d (replaying to the resume point)", loaded.seq)
    sdk._activate(session)
    saved_cwd = os.getcwd()
    os.chdir(work_dir)
    code, message = ExitCode.INFRA_ERROR, "runtime error"
    hooks = []
    try:
        session.start()
        hooks = install_hooks(session)
        if bundle.kind is JobKind.NOTEBOOK:
            from gpurelay.runtime.executor_notebook import run_notebook

            code, message = run_notebook(work_dir / bundle.entry, session)
        else:
            code, message = run_script(work_dir / bundle.entry, session)
        if worker_region is not None and code in (ExitCode.COMPLETED, ExitCode.HANDOFF):
            code, message = ExitCode.COMPLETED, "worker finished"  # WorkerDone is a Handoff
    finally:
        for hook in hooks:
            hook.remove()
        os.chdir(saved_cwd)
        sdk._activate(None)
        session.close(_STATE_FOR_EXIT[code], message)
        if code not in (ExitCode.COMPLETED, ExitCode.HANDOFF):
            # Keep the full log for diagnosis (assembly only reads committed offsets).
            try:
                storage.put_file(session.capture.log_path, segment_log_key(args.job, args.segment))
            except Exception as exc:
                log.warning("could not upload the segment log: %s", exc)
    try:
        # Libraries imported by the job (e.g. torch, and with it the GPUs) are known only now.
        final_env = env_fingerprint() | {"backend": args.backend}
        if final_env != env:
            storage.put_bytes(
                json.dumps(final_env, indent=2).encode(), segment_env_key(args.job, args.segment)
            )
    except Exception as exc:
        log.warning("could not update the segment environment: %s", exc)
    log.info("segment %s finished: exit %d (%s)", args.segment, int(code), message)
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
