"""Every GPU platform GPU Relay asks the user about (spec §21.1), in three sections.

``support`` says what GPU Relay does with a platform: ``setup`` (a free platform, connected
on the setup page), ``cloud`` (a paid platform GPU Relay starts machines on, plan G3-G5, once
the user turns it on), or ``later`` (recorded only). Credentials are kept sealed and given
only to the user's own GPU Relay, only while that platform is turned on.
"""

from __future__ import annotations

from dataclasses import dataclass

HYPERSCALER = "hyperscaler"
GPU_CLOUD = "gpu_cloud"
FREE = "free"
SECTIONS = (
    (HYPERSCALER, "Tech hyperscalers"),
    (GPU_CLOUD, "Specialised AI and ML clouds"),
    (FREE, "Free platforms"),
)


@dataclass(frozen=True)
class CredentialField:
    key: str
    label: str
    secret: bool = True  # shown as a password field
    multiline: bool = False  # a pasted file (a JSON key)
    kind: str = "text"  # "service_account": a Google Cloud service account key (JSON)
    placeholder: str = ""
    env: str = ""  # the environment variable a process acting for the user gets it in


@dataclass(frozen=True)
class Platform:
    id: str
    name: str
    section: str
    site: str
    support: str  # "setup" (free: the setup page), "cloud" (a paid backend), or "later"
    fields: tuple[CredentialField, ...] = ()
    how: str = ""  # where the user creates the credential
    note: str = ""


QUOTA = ("New accounts start with no GPU quota: request one in the {where} before GPU Relay can "
         "start GPUs there; approval can take hours to days.")  # fmt: skip

PLATFORMS: tuple[Platform, ...] = (
    Platform(
        "gcp", "Google Cloud", HYPERSCALER, "https://console.cloud.google.com", "cloud",
        (CredentialField("project_id", "Project ID", secret=False, placeholder="my-project",
                         env="GPURELAY_GCP_PROJECT"),
         CredentialField("service_account_key", "Service account key (JSON)", multiline=True,
                         kind="service_account", placeholder='{"type": "service_account", ...}',
                         env="GPURELAY_GCP_KEY")),
        how="IAM & Admin > Service Accounts: a service account only for GPU Relay, then Keys > "
            "Add key > JSON.",
        note="Free trial accounts cannot use GPUs: upgrade to a paid billing account first (unused "
             "trial credit stays until it expires). " + QUOTA.format(
                 where="console under IAM & Admin > Quotas") + " Google shows credit only in its "
             "console, so GPU Relay estimates what is left from the amount you enter.",
    ),
    Platform(
        "aws", "AWS", HYPERSCALER, "https://console.aws.amazon.com", "cloud",
        (CredentialField("access_key_id", "Access key ID", placeholder="AKIA...",
                         env="AWS_ACCESS_KEY_ID"),
         CredentialField("secret_access_key", "Secret access key", env="AWS_SECRET_ACCESS_KEY")),
        how="IAM > Users: a user only for GPU Relay, then Security credentials > Create access "
            "key.",
        note=QUOTA.format(where="Service Quotas console (EC2, On-Demand G and VT instances)"),
    ),
    Platform(
        "azure", "Azure", HYPERSCALER, "https://portal.azure.com", "cloud",
        (CredentialField("tenant_id", "Tenant (directory) ID", secret=False, env="AZURE_TENANT_ID"),
         CredentialField("client_id", "Client (application) ID", secret=False,
                         env="AZURE_CLIENT_ID"),
         CredentialField("client_secret", "Client secret", env="AZURE_CLIENT_SECRET"),
         CredentialField("subscription_id", "Subscription ID", secret=False,
                         env="AZURE_SUBSCRIPTION_ID")),
        how="Microsoft Entra ID > App registrations: an app only for GPU Relay, a client "
            "secret under Certificates & secrets, and access to one subscription.",
        note=QUOTA.format(where="portal under Quotas"),
    ),
    Platform(
        "runpod", "RunPod", GPU_CLOUD, "https://www.runpod.io", "cloud",
        (CredentialField("api_key", "API key", env="RUNPOD_API_KEY"),),
        how="RunPod console > Settings > API Keys.",
    ),
    Platform(
        "vastai", "Vast.ai", GPU_CLOUD, "https://cloud.vast.ai", "cloud",
        (CredentialField("api_key", "API key", env="VAST_API_KEY"),),
        how="Vast.ai console > Keys.",
    ),
    Platform(
        "lambda", "Lambda", GPU_CLOUD, "https://cloud.lambda.ai", "cloud",
        (CredentialField("api_key", "API key", env="LAMBDA_API_KEY"),),
        how="Lambda Cloud console > API keys.",
    ),
    Platform(
        "digitalocean", "DigitalOcean", GPU_CLOUD, "https://cloud.digitalocean.com", "cloud",
        (CredentialField("token", "Personal access token", env="DIGITALOCEAN_TOKEN"),),
        how="DigitalOcean control panel > API > Tokens > Generate New Token.",
    ),
    Platform(
        "coreweave", "CoreWeave", GPU_CLOUD, "https://www.coreweave.com", "later",
        how="",
        note="CoreWeave mostly serves companies under contract (Kubernetes). GPU Relay records "
             "that you have it and asks for access once its support is decided.",
    ),
    Platform(
        "colab", "Google Colab", FREE, "https://colab.research.google.com", "setup",
        note="GPU Relay never runs Colab for you (its rules forbid remote control): it gives "
             "you a ready notebook and you press Run all.",
    ),
    Platform(
        "kaggle", "Kaggle", FREE, "https://www.kaggle.com", "setup",
        note="About 30 free GPU hours a week, started for you through Kaggle's official API.",
    ),
    Platform(
        "modal", "Modal", FREE, "https://modal.com", "setup",
        note="About $30 of free compute a month; GPUs need a payment method on the account.",
    ),
)  # fmt: skip

BY_ID = {p.id: p for p in PLATFORMS}
SECRET_PREFIX = "platform."  # "platform.<id>.<field>": kept apart from the tokens jobs use


def secret_name(platform_id: str, field: str) -> str:
    return f"{SECRET_PREFIX}{platform_id}.{field}"


def secret_names(platform: Platform) -> list[str]:
    return [secret_name(platform.id, f.key) for f in platform.fields]


def platform_env(platform_id: str, values: dict[str, str]) -> dict[str, str]:
    """``values`` (stored names -> values) as the environment variables of one platform."""
    out = {}
    for f in BY_ID[platform_id].fields:
        value = values.get(secret_name(platform_id, f.key))
        if value and f.env:
            out[f.env] = value
    return out


def platform_secret(platform_id: str, key: str) -> str | None:
    """One credential: its environment variable if set, else the OS keyring."""
    from gpurelay.credentials import get_secret

    f = next(f for f in BY_ID[platform_id].fields if f.key == key)
    return get_secret(secret_name(platform_id, key), f.env or None)
