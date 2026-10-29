"""The "Your GPU platforms" page and its API (spec §21.1, plan G1), shared by the website and
the desktop app; only where answers and credentials are kept differs (:class:`PlatformStore`).

Nothing is ever lost by answering again: a save changes only the platforms it names, an
empty credential field keeps what is saved, and credentials go only when the user removes
that platform's. Credentials are never returned, and no process is given them before the
platform's backend exists (they are kept under their own names, apart from job tokens).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from importlib import resources
from typing import Any, Literal, Protocol

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from gpurelay.config import CLOUD_FIELDS, CloudSettings, PlatformAnswer, PlatformsSettings
from gpurelay.errors import GpuRelayError
from gpurelay.platforms.catalog import (
    BY_ID,
    PLATFORMS,
    SECTIONS,
    CredentialField,
    Platform,
    secret_name,
    secret_names,
)

VALUE_MAX = 512
FILE_MAX = 20_000  # a pasted JSON key
OTHER_MAX = 60
Answer = Literal["yes", "no", "later"]


class PlatformStore(Protocol):
    """Where one user's answers and platform credentials are kept."""

    def load(self) -> PlatformsSettings: ...
    def save(self, settings: PlatformsSettings) -> None: ...
    def saved(self) -> dict[str, float]: ...  # credential name -> when it was saved
    def put(self, values: Mapping[str, str]) -> None: ...
    def remove(self, names: Iterable[str]) -> None: ...
    def connected(self) -> dict[str, bool]: ...  # free platforms set up in GPU Relay
    def setup_link(self, platform_id: str) -> str: ...  # where a free platform is connected
    def next_path(self) -> str: ...  # where "Continue" goes
    # The paid platforms' settings (spec §21.2-21.3): None when GPU Relay is not set up yet.
    def clouds(self) -> dict[str, CloudSettings] | None: ...
    def save_clouds(self, clouds: dict[str, CloudSettings]) -> None: ...  # applies them


# The machine a paid platform starts, and where (setting name, label).
MACHINE = {
    "runpod": ("gpu", "GPU type (RunPod catalog id)"),
    "vastai": ("gpu", "GPU"),
    "lambda": ("instance_type", "Instance type"),
    "digitalocean": ("size", "GPU Droplet size"),
    "aws": ("instance_type", "Instance type"),
    "gcp": ("accelerator", "GPU"),
    "azure": ("vm_size", "VM size"),
}
REGION = {
    "lambda": ("region", "Region (empty: any with capacity)"),
    "digitalocean": ("region", "Region"),
    "aws": ("region", "Region"),
    "gcp": ("zone", "Zone"),
    "azure": ("location", "Region"),
}
NAME_OK = re.compile(r"^[A-Za-z0-9 ._:/-]{1,80}$")


class CloudUse(BaseModel):
    """How GPU Relay may use one paid platform. Fields left out stay as they are."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    terms: bool = False  # the user read the platform's terms (needed to turn it on)
    credit_usd: float | None = Field(default=None, ge=0, le=1_000_000)  # what is left now
    use_credits: bool | None = None
    card_limit_usd: float | None = Field(default=None, ge=0, le=100_000)  # a month
    machine: str | None = Field(default=None, max_length=80)
    region: str | None = Field(default=None, max_length=80)


class PlatformsForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answers: dict[str, Answer | None] = Field(default_factory=dict, max_length=50)
    credentials: dict[str, dict[str, str]] = Field(default_factory=dict, max_length=50)
    others: list[str] | None = Field(default=None, max_length=20)
    use: dict[str, CloudUse] = Field(default_factory=dict, max_length=10)
    done: bool = False  # the user finished (or skipped) the questions


class RemoveForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    platform: str = Field(max_length=40)


def _platform(platform_id: str) -> Platform:
    if platform_id not in BY_ID:
        raise GpuRelayError(f"unknown platform {platform_id!r}")
    return BY_ID[platform_id]


def _clean(field: CredentialField, value: str, platform: Platform) -> str:
    value = value.strip()
    what = f"{platform.name} {field.label.lower()}"
    if len(value) > (FILE_MAX if field.multiline else VALUE_MAX):
        raise GpuRelayError(f"the {what} is too long")
    if not field.multiline and any(c in value for c in "\r\n"):
        raise GpuRelayError(f"the {what} must be one line")
    if field.kind == "service_account":
        try:
            data = json.loads(value)
        except ValueError:
            raise GpuRelayError(f"the {what} must be the JSON key file's contents") from None
        if not (isinstance(data, dict) and data.get("type") == "service_account"
                and data.get("private_key") and data.get("client_email")):  # fmt: skip
            raise GpuRelayError(f"the {what} is not a service account key (JSON)")
    return value


def credential_values(form: PlatformsForm) -> dict[str, str]:
    """The credentials to keep, checked; empty fields are left out (the saved one stays)."""
    values: dict[str, str] = {}
    for platform_id, fields in form.credentials.items():
        platform = _platform(platform_id)
        known = {f.key: f for f in platform.fields}
        for key, value in fields.items():
            if key not in known:
                raise GpuRelayError(f"{platform.name} has no field {key!r}")
            if value.strip():
                values[secret_name(platform.id, key)] = _clean(known[key], value, platform)
    return values


def apply(settings: PlatformsSettings, form: PlatformsForm, now: float) -> PlatformsSettings:
    """``settings`` with the form's answers: only the platforms the form names change."""
    settings = settings.model_copy(deep=True)
    answers = dict(settings.answers)
    for platform_id, has in form.answers.items():
        _platform(platform_id)
        answers[platform_id] = PlatformAnswer(has=has, updated_at=now)
    for platform_id, fields in form.credentials.items():  # credentials mean an account
        if any(v.strip() for v in fields.values()) and form.answers.get(platform_id) is None:
            answers[platform_id] = PlatformAnswer(has="yes", updated_at=now)
    settings.answers = answers
    if form.others is not None:
        names: list[str] = []
        for name in form.others:
            name = " ".join(name.split())
            if len(name) > OTHER_MAX:
                raise GpuRelayError(f"a platform name is at most {OTHER_MAX} characters")
            if name and name.lower() not in {n.lower() for n in names}:
                names.append(name)
        settings.others = names
    if form.done and settings.asked_at is None:
        settings.asked_at = now
    return settings


def apply_use(
    clouds: dict[str, CloudSettings], form: PlatformsForm, saved: dict[str, float], now: float
) -> dict[str, CloudSettings]:
    """``clouds`` with the form's choices (only the platforms it names, only the fields it
    sets). Turning one on needs its terms confirmed and its credentials saved."""
    out = dict(clouds)
    for platform_id, use in form.use.items():
        platform = _platform(platform_id)
        if platform_id not in CLOUD_FIELDS:
            raise GpuRelayError(f"GPU Relay cannot start GPUs on {platform.name}")
        data = out[platform_id].model_dump()
        if use.machine is not None:
            if not NAME_OK.match(use.machine.strip()):
                raise GpuRelayError(f"{platform.name}: the machine name has odd characters")
            data[MACHINE[platform_id][0]] = use.machine.strip()
        if use.region is not None and platform_id in REGION:
            region = use.region.strip()
            if region and not NAME_OK.match(region):
                raise GpuRelayError(f"{platform.name}: the region has odd characters")
            if not region and platform_id != "lambda":
                raise GpuRelayError(f"{platform.name} needs a region")
            data[REGION[platform_id][0]] = region or None
        if use.credit_usd is not None:  # what is left now: the count starts again
            data["credit_usd"], data["credit_set_at"] = round(use.credit_usd, 2), now
        if use.use_credits is not None:
            data["use_credits"] = use.use_credits
        if use.card_limit_usd is not None:
            data["monthly_paid_cap_usd"] = round(use.card_limit_usd, 2)
        if use.terms:
            data["terms_reviewed"] = True
        if use.enabled is not None:
            if use.enabled:
                if not data["terms_reviewed"]:
                    raise GpuRelayError(f"confirm that you read {platform.name}'s terms first")
                missing = [f.label for f in platform.fields
                           if secret_name(platform_id, f.key) not in saved]  # fmt: skip
                if missing:
                    raise GpuRelayError(f"{platform.name}: save {', '.join(missing)} first")
            data["enabled"] = use.enabled
        out[platform_id] = type(out[platform_id]).model_validate(data)
    return out


def _use_view(settings: CloudSettings, platform_id: str) -> dict[str, Any]:
    machine, label = MACHINE[platform_id]
    region = REGION.get(platform_id)
    return {
        "enabled": settings.enabled, "terms": settings.terms_reviewed,
        "credit_usd": settings.credit_usd, "credit_set_at": settings.credit_set_at,
        "use_credits": settings.use_credits, "card_limit_usd": settings.monthly_paid_cap_usd,
        "machine": getattr(settings, machine), "machine_label": label,
        "region": getattr(settings, region[0]) if region else None,
        "region_label": region[1] if region else None,
        "usd_per_hour": settings.usd_per_hour,
    }  # fmt: skip


def view(store: PlatformStore) -> dict[str, Any]:
    """What the page shows: every platform, the user's answer, which credentials are saved
    (when, never their values), and whether a free platform is connected."""
    settings, saved, connected = store.load(), store.saved(), store.connected()
    clouds = store.clouds()
    platforms = []
    for p in PLATFORMS:
        answer = settings.answers.get(p.id)
        platforms.append({
            "id": p.id, "name": p.name, "section": p.section, "site": p.site,
            "support": p.support, "how": p.how, "note": p.note,
            "has": answer.has if answer else None,
            "connected": connected.get(p.id, False),
            "setup": store.setup_link(p.id) if p.support == "setup" else None,
            "use": (_use_view(clouds[p.id], p.id) if clouds is not None and p.id in clouds
                    else None),
            "fields": [{"key": f.key, "label": f.label, "secret": f.secret,
                        "multiline": f.multiline, "placeholder": f.placeholder,
                        "saved_at": saved.get(secret_name(p.id, f.key))} for f in p.fields],
        })  # fmt: skip
    return {
        "sections": [{"id": s, "title": t} for s, t in SECTIONS],
        "platforms": platforms,
        "others": settings.others,
        "asked": settings.asked_at is not None,
        "next": store.next_path(),
        "can_use": clouds is not None,  # False: finish setup first (desktop)
    }


def page() -> str:
    return resources.files("gpurelay.platforms").joinpath("page.html").read_text("utf-8")


def platforms_router(
    store_for: Callable[[Request], PlatformStore],
    *,
    action_header: str | None = None,
    too_many: Callable[[Request], bool] = lambda request: False,
    clock: Callable[[], float] = time.time,
) -> APIRouter:
    """``action_header``: a header every change must carry (the desktop app's guard against
    other local programs); the website relies on its same-origin check instead.
    ``too_many``: counts a change and says whether this user made too many (rate limit)."""
    router = APIRouter()

    def check(request: Request, value: str | None) -> None:
        if action_header is not None and value != "1":
            raise HTTPException(status_code=403, detail=f"missing {action_header} header")
        if too_many(request):
            raise HTTPException(status_code=429, detail="too many changes; try again later")

    @router.get("/platforms", response_class=HTMLResponse)
    def platforms_page() -> str:
        return page()

    @router.get("/api/platforms")
    def status(request: Request) -> dict[str, Any]:
        return view(store_for(request))

    @router.post("/api/platforms")
    def save(
        form: PlatformsForm, request: Request,
        x_gpurelay_action: str | None = Header(default=None, alias="X-GPURelay-Action"),
    ) -> dict[str, Any]:  # fmt: skip
        check(request, x_gpurelay_action)
        store = store_for(request)
        try:
            now = clock()
            values = credential_values(form)  # everything is checked before anything is kept
            settings = apply(store.load(), form, now)
            clouds = None
            if form.use:
                current = store.clouds()
                if current is None:
                    raise GpuRelayError("finish setting up GPU Relay first (where checkpoints "
                                        "live), then turn on paid platforms")  # fmt: skip
                saved = {**store.saved(), **dict.fromkeys(values, now)}
                clouds = apply_use(current, form, saved, now)
            if values:
                store.put(values)
            store.save(settings)
            if clouds is not None:
                store.save_clouds(clouds)
        except GpuRelayError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "saved": len(values), **view(store)}

    @router.post("/api/platforms/remove")
    def remove(
        form: RemoveForm, request: Request,
        x_gpurelay_action: str | None = Header(default=None, alias="X-GPURelay-Action"),
    ) -> dict[str, Any]:  # fmt: skip
        """Remove one platform's credentials (the user confirmed); the answers stay."""
        check(request, x_gpurelay_action)
        store = store_for(request)
        try:
            store.remove(secret_names(_platform(form.platform)))
            clouds = store.clouds()  # without credentials it cannot run: turn it off
            if clouds is not None and form.platform in clouds and clouds[form.platform].enabled:
                data = clouds[form.platform].model_dump() | {"enabled": False}
                clouds[form.platform] = type(clouds[form.platform]).model_validate(data)
                store.save_clouds(clouds)
        except GpuRelayError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, **view(store)}

    return router
