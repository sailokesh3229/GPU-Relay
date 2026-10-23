"""Colab backend (spec §10.5) — manual only. Colab is never automated (hard rule 2).

GPU Relay does not start, drive, or connect to Colab sessions. It only:

* writes one resume notebook per job (``resume_<job id>.ipynb``; it always resumes from
  ``LATEST.json``, so the same notebook works for every segment);
* records in storage which segment the user is asked to run (``manual_session.json``);
* lets the controller notify the user, who opens the notebook in Colab and presses Run.

The notebook reads the storage token from Colab's user secrets, installs the job's pinned
runtime wheel, refuses to run when no session is requested (or the job ran elsewhere since),
and runs the job while the user watches. Progress comes from ``status.json`` in storage.

VERIFY items (docs/verify.md): session limits, whether the deadline should count from
runtime connect rather than from "Run all".
"""

from __future__ import annotations

import json
import logging
import string
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import nbformat

from gpurelay.backends.common import session_settings
from gpurelay.config import ColabSettings, RuntimeSettings
from gpurelay.errors import GpuRelayError
from gpurelay.models import BackendCaps, BackendRunState, Job, LaunchHandle, ReadyStatus, Segment
from gpurelay.storage.base import Storage
from gpurelay.storage.layout import manual_session_key

log = logging.getLogger(__name__)

NOTEBOOK_PREFIX = "resume_"
_CELL_MARK = "# %%"


class ColabError(GpuRelayError):
    """The Colab resume notebook could not be prepared."""


def notebook_name(job_id: str) -> str:
    return f"{NOTEBOOK_PREFIX}{job_id}.ipynb"


def _cells(source: str) -> list[Any]:
    """Split a percent-format script (``# %%`` / ``# %% [markdown]``) into notebook cells."""
    cells = []
    for chunk in source.split(_CELL_MARK)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in
                     body.splitlines()]  # fmt: skip
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    return cells


def render_notebook(
    *,
    job: Job,
    repo_id: str,
    wheel: str,
    secret_name: str,
    settings: RuntimeSettings,
) -> Any:
    """The resume notebook for ``job``. Contains no secrets (the token is a Colab secret)."""
    template = string.Template(
        resources.files("gpurelay.templates").joinpath("colab_resume.py.tmpl").read_text("utf-8")
    )
    source = template.substitute(
        job_name=job.name,
        job_id=job.id,
        repo_id=repo_id,
        secret_name=secret_name,
        repo_id_py=repr(repo_id),
        job_id_py=repr(job.id),
        wheel_py=repr(wheel),
        secret_name_py=repr(secret_name),
        stale_s=repr(float(settings.stale_heartbeat_s)),
        settings_literal=repr(settings.model_dump(mode="json")),
    )
    notebook = nbformat.v4.new_notebook()
    notebook.cells = _cells(source)
    notebook.metadata.update(
        {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": notebook_name(job.id), "provenance": []},
        }
    )
    nbformat.validate(notebook)
    return notebook


@dataclass
class ColabManualBackend:
    """Prepares resume notebooks; the user runs them. ``automated`` is always False."""

    settings: ColabSettings
    runtime: RuntimeSettings
    storage: Storage
    storage_uri: str
    work_root: Path
    wheel_for: Any  # Callable[[str], str | None]: job id -> runtime wheel file name in storage
    clock: Any = time.time
    name: str = "colab_manual"
    automated: bool = False  # hard rule 2: never automated

    def capabilities(self) -> BackendCaps:
        return BackendCaps(
            gpu_types=[self.settings.gpu_label] if self.settings.gpu_label else [],
            max_session_s=int(self.settings.max_session_s),
            max_concurrent=1,
            needs_user_action=True,
        )

    def check_ready(self) -> ReadyStatus:
        if not self.storage_uri.startswith("hf:"):
            return ReadyStatus(ready=False, reason="Colab sessions need Hugging Face storage")
        return ReadyStatus(ready=True)

    def notebook_path(self, job_id: str) -> Path:
        return self.work_root / notebook_name(job_id)

    def write_notebook(self, job: Job) -> Path:
        """Write (or refresh) the job's resume notebook and return its path."""
        wheel = self.wheel_for(job.id)
        if not wheel:
            raise ColabError(f"job {job.id} has no runtime wheel in its bundle")
        notebook = render_notebook(
            job=job,
            repo_id=self.storage_uri.removeprefix("hf:"),
            wheel=wheel,
            secret_name=self.settings.secret_name,
            settings=session_settings(self.runtime),
        )
        path = self.notebook_path(job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, str(path))
        return path

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        """Prepare the notebook and ask for a session; nothing is started here."""
        path = self.write_notebook(job)
        request = {
            "segment_id": segment.id,
            "deadline_s": float(min(deadline_s, self.settings.max_session_s)),
            "start_checkpoint_seq": segment.start_checkpoint_seq,
            "requested_at": self.clock(),
        }
        self.storage.put_bytes(json.dumps(request).encode(), manual_session_key(job.id))
        return LaunchHandle(
            backend=self.name,
            handle_id=str(path),
            extra={"job_id": job.id, "segment_id": segment.id, "notebook": str(path)},
        )

    def poll(self, handle: LaunchHandle) -> BackendRunState:
        # GPU Relay cannot see Colab; progress and endings come from status.json.
        return BackendRunState.UNKNOWN

    def cancel(self, handle: LaunchHandle) -> None:
        """Withdraw the request so the notebook refuses to start this segment."""
        job_id = handle.extra.get("job_id")
        if not job_id:
            return
        key = manual_session_key(job_id)
        raw = self.storage.get_bytes(key)
        if raw is not None and json.loads(raw).get("segment_id") == handle.extra.get("segment_id"):
            self.storage.delete(key)
        # A session the user already started stops through control.json (next safe point).

    def fetch_console_log(self, handle: LaunchHandle) -> str | None:
        return None  # the notebook's output is in Colab; the job log is in storage
