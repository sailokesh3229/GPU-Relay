"""Remote key layout inside storage (spec §11.3)."""

from __future__ import annotations

import re

from gpurelay.constants import JOBS_PREFIX, LATEST_FILENAME, STATUS_FILENAME

CKPT_DIR_RE = re.compile(r"^ckpt-(\d+)$")
MANIFEST_FILENAME = "manifest.json"


def job_prefix(job_id: str) -> str:
    return f"{JOBS_PREFIX}/{job_id}"


def checkpoints_prefix(job_id: str) -> str:
    return f"{job_prefix(job_id)}/checkpoints"


def ckpt_dirname(seq: int) -> str:
    return f"ckpt-{seq}"


def ckpt_prefix(job_id: str, seq: int) -> str:
    return f"{checkpoints_prefix(job_id)}/{ckpt_dirname(seq)}"


def latest_key(job_id: str) -> str:
    return f"{checkpoints_prefix(job_id)}/{LATEST_FILENAME}"


def manifest_key(job_id: str, seq: int) -> str:
    return f"{ckpt_prefix(job_id, seq)}/{MANIFEST_FILENAME}"


def status_key(job_id: str) -> str:
    return f"{job_prefix(job_id)}/{STATUS_FILENAME}"


def control_key(job_id: str) -> str:
    """Controller -> runtime requests (e.g. ``{"handoff": true}`` to pause)."""
    return f"{job_prefix(job_id)}/control.json"


def colab_notebook_key(job_id: str) -> str:
    """The job's Colab resume notebook, for the launcher notebook (``colab/open_job.ipynb``)."""
    return f"{job_prefix(job_id)}/colab/resume.ipynb"


def manual_session_key(job_id: str) -> str:
    """Controller -> manual (Colab) notebook: which segment the user is asked to run."""
    return f"{job_prefix(job_id)}/manual_session.json"


def parallel_prefix(job_id: str, region: str) -> str:
    """Shared state of a parallel chunks region: ``region.json``, ``claims/``, ``done/``."""
    return f"{job_prefix(job_id)}/parallel/{region}"


def worker_key(job_id: str, segment_id: str) -> str:
    """A worker segment's assignment (written by the controller before launch)."""
    return f"{job_prefix(job_id)}/workers/{segment_id}.json"


def worker_status_key(job_id: str, segment_id: str) -> str:
    """A worker segment's heartbeat (workers never write the job's ``status.json``)."""
    return f"{job_prefix(job_id)}/workers/{segment_id}.status.json"


def blob_key(job_id: str, sha256: str) -> str:
    """Content-addressed immutable object (figures, chunk results, working files)."""
    return f"{job_prefix(job_id)}/blobs/{sha256}"


def bundle_prefix(job_id: str) -> str:
    return f"{job_prefix(job_id)}/bundle"


def segment_prefix(job_id: str, segment_id: str) -> str:
    return f"{job_prefix(job_id)}/segments/{segment_id}"


def segment_log_key(job_id: str, segment_id: str) -> str:
    return f"{segment_prefix(job_id, segment_id)}/log.txt"


def segment_env_key(job_id: str, segment_id: str) -> str:
    return f"{segment_prefix(job_id, segment_id)}/env.json"


def final_prefix(job_id: str) -> str:
    return f"{job_prefix(job_id)}/final"


def parse_ckpt_seq(key: str, job_id: str) -> int | None:
    """Return the checkpoint seq of a key under ``checkpoints/ckpt-<seq>/``, else None."""
    prefix = checkpoints_prefix(job_id) + "/"
    if not key.startswith(prefix):
        return None
    first = key[len(prefix) :].split("/", 1)[0]
    match = CKPT_DIR_RE.match(first)
    return int(match.group(1)) if match else None
