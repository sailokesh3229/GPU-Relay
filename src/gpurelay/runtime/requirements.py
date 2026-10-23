"""Install a job's requirements inside a cloud session (spec §12.1 step 2).

Platform images ship CUDA builds of the big frameworks. They are **not** reinstalled unless
the user pinned a different major.minor version (``torch==2.3.1`` on an image with 2.4.0);
everything else is installed with pip. Differences are logged so they can be compared
across segments (``segments/<seg>/env.json`` records the versions actually used).
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PREINSTALLED_FRAMEWORKS = frozenset(
    {"torch", "torchvision", "torchaudio", "tensorflow", "tensorflow-gpu", "jax", "jaxlib",
     "keras", "tensorflow-cpu"}
)  # fmt: skip
_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")
_PIN = re.compile(r"==\s*([0-9]+)\.([0-9]+)")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Plan:
    install: list[str]
    kept: list[str]  # preinstalled frameworks left alone


def plan_requirements(lines: list[str]) -> Plan:
    """Decide which requirement lines to pip-install."""
    install: list[str] = []
    kept: list[str] = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):  # pip options (-r, --index-url, ...): pass through
            install.append(line)
            continue
        match = _REQ.match(line.split(";", 1)[0])
        if not match:
            install.append(line)
            continue
        name = _canonical(match.group(1))
        if name in PREINSTALLED_FRAMEWORKS:
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                install.append(line)
                continue
            pin = _PIN.search(match.group(3) or "")
            have = installed.split(".")[:2]
            if pin and [pin.group(1), pin.group(2)] != have:
                log.warning("%s: image has %s, job pins %s -> reinstalling", name, installed, line)
                install.append(line)
            else:
                if pin is None or match.group(3).strip() != f"=={installed}":
                    log.info("%s: keeping the image's %s (job asks %s)", name, installed, line)
                kept.append(line)
            continue
        install.append(line)
    return Plan(install=install, kept=kept)


def install_requirements(path: Path) -> Plan:
    """pip-install ``path`` following the policy above."""
    plan = plan_requirements(Path(path).read_text(encoding="utf-8").splitlines())
    if plan.install:
        args = []
        for line in plan.install:
            args.extend(line.split() if line.startswith("-") else [line])
        log.info("installing %d requirement(s)", len(plan.install))
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                *args,
            ]
        )
    return plan
