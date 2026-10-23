"""Final output assembly (spec §9).

The final checkpoint's ``runtime.chain`` lists, for every segment that contributed,
how many log bytes and which figures were committed. Each segment's log is truncated
at its committed offset (work after it was redone by the next segment) and the parts
are concatenated; figures are merged in the same order.

Output (locally, and under ``jobs/<job>/final/`` in storage)::

    log.txt  figures/0000.png ...  files/<path>  report.json
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpurelay.bundle.packager import read_bundle_file, read_manifest
from gpurelay.errors import GpuRelayError
from gpurelay.models import utcnow
from gpurelay.runtime.blobs import BlobCache
from gpurelay.runtime.checkpoint import load_latest_checkpoint
from gpurelay.storage.base import Storage
from gpurelay.storage.layout import final_prefix, segment_env_key, segment_log_key


class AssemblyError(GpuRelayError):
    """Final outputs cannot be assembled (job not completed or outputs missing)."""


@dataclass
class AssemblyResult:
    out_dir: Path
    segments: list[str] = field(default_factory=list)
    log_bytes: int = 0
    figures: int = 0
    files: list[str] = field(default_factory=list)


def _write_output_notebook(
    storage: Storage, job_id: str, cells: dict[str, str], blobs: BlobCache, out_dir: Path
) -> None:
    """``output.ipynb``: the submitted notebook with every cell's recorded outputs."""
    import nbformat

    from gpurelay.runtime.executor_notebook import build_output_notebook

    manifest = read_manifest(storage, job_id)
    source = nbformat.reads(read_bundle_file(storage, job_id, manifest.entry).decode("utf-8"), 4)
    outputs = {int(k): json.loads(blobs.read_bytes(sha)) for k, sha in cells.items()}
    nbformat.write(build_output_notebook(source, outputs), str(out_dir / "output.ipynb"))


def assemble(
    storage: Storage, job_id: str, out_dir: Path, *, upload: bool = True
) -> AssemblyResult:
    """Build the job's final outputs into ``out_dir`` (and upload them to storage)."""
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory(prefix="gpurelay-assemble-") as tmp:
        loaded = load_latest_checkpoint(storage, job_id, Path(tmp) / "ckpt")
        if loaded is None or not loaded.runtime.get("final"):
            raise AssemblyError(f"job {job_id} has no final checkpoint (not completed)")
        runtime = loaded.runtime
        blobs = BlobCache(Path(tmp) / "blobs", storage, job_id)

        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "figures").mkdir(parents=True)
        (out_dir / "files").mkdir()
        result = AssemblyResult(out_dir=out_dir)
        report_segments: list[dict[str, Any]] = []

        with open(out_dir / "log.txt", "wb") as merged:
            for entry in runtime["chain"]:
                segment = entry["segment"]
                n = int(entry["log_bytes"])
                data = storage.get_bytes(segment_log_key(job_id, segment)) or b""
                if len(data) < n:
                    raise AssemblyError(
                        f"log of segment {segment} has {len(data)} bytes, {n} committed"
                    )
                merged.write(data[:n])
                for sha in entry["figures"]:
                    blobs.copy_to(sha, out_dir / "figures" / f"{result.figures:04d}.png")
                    result.figures += 1
                result.log_bytes += n
                result.segments.append(segment)
                env_raw = storage.get_bytes(segment_env_key(job_id, segment))
                report_segments.append(
                    {
                        "segment_id": segment,
                        "log_bytes": n,
                        "figures": len(entry["figures"]),
                        "env": json.loads(env_raw) if env_raw else None,
                    }
                )

        for rel, sha in sorted(runtime.get("files", {}).items()):
            blobs.copy_to(sha, out_dir / "files" / rel)
            result.files.append(rel)

        report = {
            "job_id": job_id,
            "assembled_at": utcnow().isoformat(),
            "final_checkpoint_seq": loaded.seq,
            "segments": report_segments,
            "gpu_models": sorted(
                {gpu for s in report_segments if s["env"] for gpu in s["env"].get("gpus", [])}
            ),
            "log_bytes": result.log_bytes,
            "figures": result.figures,
            "files": result.files,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if "cells" in runtime:
            _write_output_notebook(storage, job_id, runtime["cells"], blobs, out_dir)

    if upload:
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(out_dir).as_posix()
            storage.put_file(path, f"{final_prefix(job_id)}/{rel}")
    return result
