"""User notifications (spec §13.6): desktop + optional ntfy.sh topic.

Events: Colab action needed, job completed, job failed, waiting for quota, budget low.
Delivery failures are logged, never raised.
"""

from __future__ import annotations

import importlib.util
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from gpurelay.config import NotifySettings

log = logging.getLogger(__name__)

NTFY_URL = "https://ntfy.sh"


@dataclass
class Notification:
    kind: str  # colab_action | completed | failed | waiting_for_quota | budget_low |
    # controller_error
    title: str
    message: str


@dataclass
class Notifier:
    settings: NotifySettings
    sent: list[Notification] = field(default_factory=list)
    _post: Callable[[str, bytes, dict[str, str]], None] | None = None

    def notify(self, kind: str, title: str, message: str) -> None:
        note = Notification(kind, title, message)
        self.sent.append(note)
        log.info("notify [%s] %s: %s", kind, title, message)
        if self.settings.desktop:
            self._desktop(note)
        if self.settings.ntfy_topic:
            self._ntfy(note)

    def _desktop(self, note: Notification) -> None:
        if importlib.util.find_spec("plyer") is None:
            return
        try:
            from plyer import notification

            notification.notify(title=f"GPU Relay: {note.title}", message=note.message[:250])
        except Exception as exc:  # no display, missing backend, ...
            log.debug("desktop notification failed: %s", exc)

    def _ntfy(self, note: Notification) -> None:
        url = f"{NTFY_URL}/{self.settings.ntfy_topic}"
        headers = {"Title": f"GPU Relay: {note.title}", "Tags": note.kind}
        try:
            post = self._post or _http_post
            post(url, note.message.encode("utf-8"), headers)
        except Exception as exc:
            log.warning("ntfy notification failed: %s", exc)


def _http_post(url: str, body: bytes, headers: dict[str, str]) -> None:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=10):
        pass
