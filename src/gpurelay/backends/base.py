"""Backend interface (spec §10.1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gpurelay.models import BackendCaps, BackendRunState, Job, LaunchHandle, ReadyStatus, Segment


@runtime_checkable
class Backend(Protocol):
    """A compute platform that runs job segments.

    Job progress is read from ``status.json`` in storage (the source of truth); the
    backend's own state is used to detect crashed and finished sessions.
    """

    name: str
    automated: bool

    def capabilities(self) -> BackendCaps:
        """GPU types, max session length, concurrency, whether the user must act."""
        ...

    def check_ready(self) -> ReadyStatus:
        """Credentials present, API reachable."""
        ...

    def launch(self, job: Job, segment: Segment, deadline_s: float) -> LaunchHandle:
        """Start a segment that must hand off within ``deadline_s`` seconds."""
        ...

    def poll(self, handle: LaunchHandle) -> BackendRunState:
        """queued | running | finished | error | unknown."""
        ...

    def cancel(self, handle: LaunchHandle) -> None:
        """Stop the session."""
        ...

    def fetch_console_log(self, handle: LaunchHandle) -> str | None:
        """Raw console output of the session, if available."""
        ...
