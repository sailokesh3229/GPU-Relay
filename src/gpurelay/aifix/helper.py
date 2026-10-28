"""``gpurelay helper``: the laptop side of "Fix it with AI" for the website (spec §20.2).

The website cannot reach the user's laptop, so the helper asks it for work: every few seconds
it tells the site it is there (and which AI tools this computer has), and when a fix request
is waiting it asks the AI here (Claude Code or Codex, under the user's own plan, in the order
the site chose) and posts the answer.
Only outgoing HTTPS; nothing listens on this computer.

The site's address is saved in the GPU Relay home (``helper.json``) and the helper key in the
OS keyring, so after the first run ``gpurelay helper`` alone starts it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpurelay import __version__
from gpurelay.aifix import choice, claude_code
from gpurelay.aifix.request import FixRequest
from gpurelay.aifix.tool import ToolInfo

POLL_S = 5.0
MAX_BACKOFF_S = 60.0
DETECT_EVERY_S = 300.0
SETTINGS_NAME = "helper.json"
KEY_SECRET = "helper_key"
ENV_KEY = "GPURELAY_HELPER_KEY"


class HelperRefused(Exception):
    """The site refused the helper (wrong key); retrying will not help."""


class SiteClient:
    def __init__(self, site: str, key: str, timeout: float = 90.0) -> None:
        self.site = site.rstrip("/")
        self.key = key
        self.timeout = timeout

    def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.site + path, data=json.dumps(data).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "User-Agent": f"gpurelay-helper/{__version__}"},
        )  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                answer = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = _detail(exc)
            if exc.code == 401:
                raise HelperRefused(detail) from None
            raise ConnectionError(f"{exc.code}: {detail}") from None
        return answer if isinstance(answer, dict) else {}


def _detail(exc: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(exc.read()).get("detail") or exc.reason)
    except Exception:
        return str(exc.reason)


def load_site(home: Path) -> str | None:
    try:
        return str(json.loads((home / SETTINGS_NAME).read_text("utf-8"))["site"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_site(home: Path, site: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / SETTINGS_NAME).write_text(json.dumps({"site": site}), encoding="utf-8")


def run(
    client: SiteClient,
    *,
    model: str | None = None,
    once: bool = False,
    detect: Callable[[], list[ToolInfo]] = choice.detect_all,
    solver: Callable[..., dict[str, Any]] = choice.solve,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print,
) -> int:
    """Answer fix requests until stopped (``once``: one round, for tests and checks).
    ``model``: the Claude model to use instead of the site's choice."""
    tools: list[dict[str, Any]] = []
    checked: float | None = None
    backoff = POLL_S
    connected = False
    while True:
        if checked is None or clock() - checked > DETECT_EVERY_S:
            tools = [t.to_json() for t in detect()]
            checked = clock()
            ready = [t["label"] for t in tools if t["signed_in"]]
            if ready:
                say(f"AI tools here: {', '.join(ready)}.")
            elif tools:
                say(f"{tools[0]['label']} is installed but not signed in (Claude Code: run "
                    "`claude` and /login; Codex: `codex login`).")  # fmt: skip
            else:
                say("No AI tool here: install Claude Code (or Codex) and sign in, then start "
                    "the helper again.")  # fmt: skip
        try:
            answer = client.post("/helper/v1/poll", {"tools": tools, "version": __version__})
        except HelperRefused as exc:
            say(f"The site refused this helper: {exc}")
            return 2
        except (OSError, ConnectionError, ValueError) as exc:
            say(f"Cannot reach {client.site} ({exc}); trying again in {round(backoff)} s.")
            connected = False
            if once:
                return 1
            sleep(backoff)
            backoff = min(MAX_BACKOFF_S, backoff * 2)
            continue
        backoff = POLL_S
        if not connected:
            connected = True
            say(f"Connected to {client.site}. Waiting for fix requests (Ctrl+C to stop).")
        fix = answer.get("fix")
        if isinstance(fix, dict):
            available = {t["name"] for t in tools if t["signed_in"]}
            _answer(client, fix, model, available, solver, say)
            continue  # there may be another one waiting
        if once:
            return 0
        sleep(POLL_S)


def _answer(client: SiteClient, fix: dict[str, Any], model: str | None, available: set[str],
            solver: Callable[..., dict[str, Any]], say: Callable[[str], None]) -> None:  # fmt: skip
    req = FixRequest.from_json(fix["request"])
    steps = fix.get("plan") or [{"tool": claude_code.NAME, "model": fix.get("model")}]
    if model:  # the user's choice for Claude Code
        steps = [{**s, "model": model} if s.get("tool") == claude_code.NAME else s for s in steps]
    names = ", ".join(f"{s.get('tool')}/{s.get('model') or 'default'}" for s in steps)
    say(f"Fixing {req.name} ({req.error[:100]}); trying {names}")
    result = solver(req, steps, available)
    if result.get("ok"):
        say(f"  {result.get('tool')}/{result.get('model')} proposed "
            f"{len(result.get('changes', []))} change(s) in {result.get('seconds')} s")  # fmt: skip
    else:
        say(f"  no fix: {result.get('error')}")
    _send(client, str(fix["id"]), result, say)


def _send(client: SiteClient, fix_id: str, result: dict[str, Any],
          say: Callable[[str], None]) -> None:  # fmt: skip
    try:
        client.post(f"/helper/v1/fixes/{fix_id}", result)
    except (OSError, ConnectionError, ValueError, HelperRefused) as exc:
        say(f"  the answer could not be sent ({exc}); ask again on the job page")
