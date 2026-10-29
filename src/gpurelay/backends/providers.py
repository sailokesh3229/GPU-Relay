"""The provider behind each paid platform (spec §21), with its credentials: the platform's
environment variables when set (the website gives a user's own GPU Relay only those of the
platforms they turned on), else the OS keyring (desktop)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gpurelay.backends.cloud import CloudProvider
from gpurelay.config import CloudSettings


def make_provider(
    platform_id: str,
    settings: CloudSettings,
    secret: Callable[[str], str | None] | None = None,
    **kw: Any,
) -> CloudProvider:
    """``secret``: field key -> value (default: environment, then keyring)."""
    if secret is None:
        from gpurelay.platforms.catalog import platform_secret

        def secret(key: str) -> str | None:
            return platform_secret(platform_id, key)

    provider: Any
    if platform_id == "runpod":
        from gpurelay.backends.runpod import RunPodProvider

        provider = RunPodProvider(settings, secret, **kw)  # type: ignore[arg-type]
    elif platform_id == "vastai":
        from gpurelay.backends.vast import VastProvider

        provider = VastProvider(settings, secret, **kw)  # type: ignore[arg-type]
    elif platform_id == "lambda":
        from gpurelay.backends.lambdacloud import LambdaProvider

        provider = LambdaProvider(settings, secret, **kw)  # type: ignore[arg-type]
    elif platform_id == "digitalocean":
        from gpurelay.backends.digitalocean import DigitalOceanProvider

        provider = DigitalOceanProvider(settings, secret, **kw)  # type: ignore[arg-type]
    elif platform_id == "aws":
        from gpurelay.backends.aws import AwsProvider

        provider = AwsProvider(settings, secret, **kw)  # type: ignore[arg-type]
    elif platform_id == "gcp":
        from gpurelay.backends.gcp import GcpProvider

        provider = GcpProvider(settings, secret, **kw)  # type: ignore[arg-type]
    elif platform_id == "azure":
        from gpurelay.backends.azure import AzureProvider

        provider = AzureProvider(settings, secret, **kw)  # type: ignore[arg-type]
    else:
        raise ValueError(f"no paid backend for {platform_id!r}")
    result: CloudProvider = provider
    return result
