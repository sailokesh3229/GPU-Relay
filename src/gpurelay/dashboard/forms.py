"""Request bodies and small computations for the web app's write screens (plan W2).

Uploads come as JSON with base64 file contents (no multipart dependency). Files are written
under a fresh temporary directory with a checked name, never a path from the request.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpurelay.controller.db import LedgerEntry

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # the script or notebook, decoded
MAX_BODY_BYTES = MAX_UPLOAD_BYTES * 4 // 3 + 64 * 1024  # base64 plus the other fields
FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
DAY_S = 86400.0


class FormError(ValueError):
    """A bad request body: shown to the user as is."""


class UploadForm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    content: str  # base64
    requirements: str | None = Field(default=None, max_length=64 * 1024)  # requirements.txt text


class SubmitForm(UploadForm):
    name: str | None = Field(default=None, max_length=100)
    priority: int = Field(default=0, ge=-1000, le=1000)
    estimate: str | None = Field(default=None, max_length=20)
    data: list[str] = Field(default_factory=list, max_length=20)
    prefer: str | None = None
    accept_restart_only: bool = False


class QuotaForm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    used: str | None = Field(default=None, max_length=20)
    remaining: str | None = Field(default=None, max_length=20)
    # In the platform's own unit (dollars, credits) as its billing page shows them (P11).
    used_amount: float | None = Field(default=None, ge=0, le=1e6)
    remaining_amount: float | None = Field(default=None, ge=0, le=1e6)


def parse_body(raw: bytes, model: type[BaseModel]) -> Any:
    if len(raw) > MAX_BODY_BYTES:
        raise FormError(f"the upload is too large (limit {MAX_UPLOAD_BYTES // 2**20} MB)")
    try:
        return model.model_validate(json.loads(raw or b"{}"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FormError("the request body is not JSON") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "body"
        raise FormError(f"{where}: {first['msg']}") from exc


@dataclass
class SavedUpload:
    path: Path
    requirements: Path | None
    lines: int


def save_upload(form: UploadForm, workdir: Path) -> SavedUpload:
    """Write the uploaded file (and requirements) into ``workdir`` after checking them."""
    name = form.filename.strip()
    if not FILENAME_RE.match(name) or Path(name).suffix not in (".py", ".ipynb"):
        raise FormError("choose a Python script (.py) or a notebook (.ipynb) with a plain name")
    try:
        data = base64.b64decode(form.content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FormError("the file content is not valid base64") from exc
    if not data:
        raise FormError(f"{name} is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise FormError(f"{name} is too large (limit {MAX_UPLOAD_BYTES // 2**20} MB)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormError(f"{name} is not UTF-8 text") from exc
    path = workdir / name
    path.write_bytes(data)
    requirements = None
    if form.requirements and form.requirements.strip():
        requirements = workdir / "requirements.txt"
        requirements.write_text(form.requirements, encoding="utf-8")
    return SavedUpload(path, requirements, text.count("\n") + (not text.endswith("\n")))


def daily_usage(entries: Iterable[LedgerEntry], now: float, days: int = 7) -> list[float]:
    """GPU hours per UTC day for the last ``days`` days, oldest first (today last)."""
    today = now - now % DAY_S
    starts = [today - DAY_S * (days - 1 - i) for i in range(days)]
    out = [0.0] * days
    for e in entries:
        if e.kind != "usage" or e.start_ts is None or e.end_ts is None:
            continue
        for i, day in enumerate(starts):
            overlap = min(e.end_ts, day + DAY_S, now) - max(e.start_ts, day)
            if overlap > 0:
                out[i] += overlap / 3600
    return [round(h, 3) for h in out]
