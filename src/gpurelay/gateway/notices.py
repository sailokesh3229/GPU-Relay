"""Job emails (plan M4): a job needs the user on Colab, finished, or failed.

The gateway looks at each running user's jobs every half minute (``Children.tick``) and
mails a user when one of their jobs enters one of those states. The first look after the
site starts only learns the states, so a restart sends nothing twice. Users without a
verified email (when the site sends mail) and users who turned notices off get none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gpurelay.accounts.store import UserStore
from gpurelay.constants import COLAB_LAUNCHER_URL
from gpurelay.gateway.mail import Mailer

MESSAGES = {
    "awaiting_user": ("needs you on Colab", "Open it in Colab with this link, choose a GPU "
                      f"(Runtime > Change runtime type) and press Run all:\n{COLAB_LAUNCHER_URL}"),
    "completed": ("finished", "Its outputs are ready in your GPU Relay."),
    "failed_user": ("stopped with an error in the script", "The job page shows where it failed. "
                    "Check it yourself there, or click Fix it with AI."),
    "failed_infra": ("failed on the platforms", "The job page shows what happened; you can "
                     "retry it there."),
}  # fmt: skip


@dataclass
class Notices:
    store: UserStore
    mailer: Mailer | None
    _seen: dict[str, dict[str, str]] = field(default_factory=dict)  # user -> job id -> state

    def __call__(self, user_id: str, jobs: list[dict[str, Any]]) -> list[str]:
        """Mail what changed since the last look; returns the subjects sent (for tests)."""
        states = {str(j["id"]): str(j["state"]) for j in jobs if "id" in j}
        before = self._seen.get(user_id)
        self._seen[user_id] = states
        if before is None or self.mailer is None:
            return []
        user = self.store.get(user_id)
        if user is None or not user.notify or not user.email_verified:
            return []
        sent = []
        for job in jobs:
            job_id, state = str(job.get("id")), str(job.get("state"))
            if state not in MESSAGES or before.get(job_id) == state:
                continue
            what, hint = MESSAGES[state]
            subject = f"GPU Relay: {job.get('name') or job_id} {what}"
            link = self.mailer.link(f"/app/#/job/{job_id}")
            body = f"Hello,\n\nyour job {job.get('name') or job_id} {what}.\n{hint}\n\n{link}\n"
            if self.mailer.send(user.email, subject, body):
                sent.append(subject)
        return sent
