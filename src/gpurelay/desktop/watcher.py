"""Turns job state changes (from the web app's ``/api/overview``) into notifications.

The first snapshot only primes the watcher: starting the app never replays old events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# state -> (notification kind, title template, message template)
_EVENTS: dict[str, tuple[str, str, str]] = {
    "completed": ("completed", "{name} finished", "Outputs are ready. {used} of GPU time."),
    "failed_user": ("failed", "{name} failed", "Your code raised an error: {message}"),
    "failed_infra": ("failed", "{name} stopped", "A platform problem stopped it: {message}"),
    "awaiting_user": (
        "colab_action",
        "{name} needs you on Colab",
        "Save the resume notebook from the job page, open it in Colab and press Run all.",
    ),
    "waiting_for_quota": (
        "waiting_for_quota",
        "{name} is waiting for GPU hours",
        "It continues automatically when the weekly hours free up.",
    ),
}


@dataclass(frozen=True)
class Note:
    kind: str
    title: str
    message: str
    job_id: str


def _hours(h: float | None) -> str:
    if not h:
        return "no time"
    return f"{round(h * 60)} min" if h < 1 else f"{h:.1f} h"


@dataclass
class JobWatcher:
    states: dict[str, str] | None = None  # job id -> last state; None until primed
    active: int = 0
    needs_you: int = 0

    def update(self, overview: dict[str, Any]) -> list[Note]:
        """Notes for jobs whose state changed into a notable one since the last snapshot."""
        jobs = overview.get("jobs") or []
        current = {str(j["id"]): str(j["state"]) for j in jobs if "id" in j and "state" in j}
        self.active = sum(1 for j in jobs if not j.get("terminal"))
        self.needs_you = sum(1 for j in jobs if j.get("state") == "awaiting_user")
        primed, self.states = self.states, current
        if primed is None:
            return []
        notes = []
        for job in jobs:
            state = current.get(str(job.get("id")))
            if state is None or primed.get(str(job["id"])) == state or state not in _EVENTS:
                continue
            kind, title, message = _EVENTS[state]
            values = {
                "name": job.get("name") or job["id"],
                "used": _hours(job.get("used_h")),
                "message": (job.get("message") or "see the job page")[:160],
            }
            notes.append(Note(kind, title.format(**values), message.format(**values), job["id"]))
        return notes

    def summary(self) -> str:
        """One line for the tray tooltip."""
        if self.states is None:
            return "GPU Relay"
        parts = [f"{self.active} job{'s' if self.active != 1 else ''} active"]
        if self.needs_you:
            parts.append(f"{self.needs_you} need{'s' if self.needs_you == 1 else ''} you")
        return "GPU Relay: " + ", ".join(parts)
