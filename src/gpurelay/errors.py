"""Exception hierarchy for GPU Relay."""

from __future__ import annotations


class GpuRelayError(Exception):
    """Base class for all GPU Relay errors."""


class ConfigError(GpuRelayError):
    """The configuration file is missing, malformed, or invalid."""


class HardRuleViolation(GpuRelayError):
    """An operation would violate a hard rule from spec §3 (platform terms)."""


class StorageError(GpuRelayError):
    """A storage operation failed."""


class StorageNotFoundError(StorageError):
    """The requested remote key does not exist."""


class InvalidRemotePathError(GpuRelayError, ValueError):
    """A remote key is malformed (absolute, contains '..', backslashes, etc.)."""


class ChecksumMismatchError(StorageError):
    """Stored bytes do not match the expected SHA-256."""


class CheckpointError(GpuRelayError):
    """Base class for checkpoint failures."""


class CheckpointCorruptError(CheckpointError):
    """No committed checkpoint could be loaded and verified (all candidates failed)."""


class StateRestoreError(CheckpointError):
    """Saved variables could not be restored into the namespace."""


class BackendBusyError(GpuRelayError):
    """The backend refused a launch because the job already has a session there."""


class ControllerRunningError(GpuRelayError):
    """Another ``gpurelay up`` already holds the controller lock for this home."""


class NotYetImplementedError(GpuRelayError):
    """A feature scheduled for a later implementation phase was invoked."""

    def __init__(self, feature: str, phase: int) -> None:
        super().__init__(f"{feature}: not yet implemented (phase {phase})")
        self.feature = feature
        self.phase = phase
