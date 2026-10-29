"""Email from the website (plan M4): verification, password reset, job notices.

Optional. With ``GPURELAY_BREVO_API_KEY`` set the site sends mail through Brevo's HTTPS API;
else with ``GPURELAY_SMTP_HOST`` set, through that SMTP server (Gmail with an app password,
Mailgun, ...). New accounts then verify their email before they connect platforms, and
"forgot password" works. Without either there is no email at all: accounts need no
verification and a forgotten password is reset with ``gpurelay reset-link``.

Render's free plan blocks outgoing SMTP (ports 25, 465, 587), so on it use Brevo: its free
plan sends 300 mails a day over HTTPS. ``GPURELAY_MAIL_FROM`` must be a sender verified in
Brevo (Senders, domains & dedicated IPs > Senders).

* ``GPURELAY_BREVO_API_KEY`` (Brevo > SMTP & API > API keys, ``xkeysib-...``);
* ``GPURELAY_SMTP_HOST``, ``GPURELAY_SMTP_PORT`` (587: STARTTLS; 465: TLS);
* ``GPURELAY_SMTP_USER``, ``GPURELAY_SMTP_PASSWORD``;
* ``GPURELAY_MAIL_FROM`` (default: the SMTP user);
* ``GPURELAY_SITE_URL``: the site's address for links in mails (``https://...``).
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import EmailMessage

from gpurelay.errors import ConfigError

log = logging.getLogger(__name__)

ENV_HOST = "GPURELAY_SMTP_HOST"
ENV_PORT = "GPURELAY_SMTP_PORT"
ENV_USER = "GPURELAY_SMTP_USER"
ENV_PASSWORD = "GPURELAY_SMTP_PASSWORD"
ENV_FROM = "GPURELAY_MAIL_FROM"
ENV_SITE_URL = "GPURELAY_SITE_URL"
ENV_BREVO = "GPURELAY_BREVO_API_KEY"
BREVO_URL = "https://api.brevo.com/v3/smtp/email"


@dataclass
class Mailer:
    host: str
    port: int
    user: str | None
    password: str | None
    sender: str
    site_url: str
    brevo_key: str | None = None  # set: Brevo's HTTPS API instead of SMTP

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Mailer | None:
        env = os.environ if env is None else env
        host = env.get(ENV_HOST, "").strip()
        brevo = env.get(ENV_BREVO, "").strip()
        if not host and not brevo:
            return None
        site = env.get(ENV_SITE_URL, "").strip().rstrip("/")
        if not site.startswith(("https://", "http://")):
            raise ConfigError(f"set {ENV_SITE_URL} (https://...) so mails can link to the site")
        user = env.get(ENV_USER, "").strip() or None
        sender = env.get(ENV_FROM, "").strip() or user
        if not sender:
            raise ConfigError(f"set {ENV_FROM}: the address mails come from")
        if brevo:
            return cls("api.brevo.com", 443, None, None, sender, site, brevo_key=brevo)
        try:
            port = int(env.get(ENV_PORT, "") or 587)
        except ValueError:
            raise ConfigError(f"{ENV_PORT} must be a number") from None
        return cls(host, port, user, env.get(ENV_PASSWORD) or None, sender, site)

    def link(self, path: str) -> str:
        return self.site_url + path

    def send(self, to: str, subject: str, body: str) -> bool:
        """Send one plain-text mail; False (logged, never raised) if the server refused it."""
        if self.brevo_key:
            return self._send_brevo(to, subject, body)
        message = EmailMessage()
        message["From"], message["To"], message["Subject"] = self.sender, to, subject
        message.set_content(body)
        context = ssl.create_default_context()
        try:
            if self.port == 465:
                with smtplib.SMTP_SSL(self.host, self.port, context=context, timeout=20) as smtp:
                    self._deliver(smtp, message)
            else:
                with smtplib.SMTP(self.host, self.port, timeout=20) as smtp:
                    smtp.starttls(context=context)
                    self._deliver(smtp, message)
        except (OSError, smtplib.SMTPException) as exc:
            log.warning("mail to a user failed: %s", type(exc).__name__)
            return False
        return True

    def _send_brevo(self, to: str, subject: str, body: str) -> bool:
        payload = {"sender": {"name": "GPU Relay", "email": self.sender}, "to": [{"email": to}],
                   "subject": subject, "textContent": body}  # fmt: skip
        request = urllib.request.Request(
            BREVO_URL, data=json.dumps(payload).encode(), method="POST",
            headers={"api-key": self.brevo_key or "", "Content-Type": "application/json",
                     "Accept": "application/json"},
        )  # fmt: skip
        try:
            with urllib.request.urlopen(request, timeout=20):
                return True
        except urllib.error.HTTPError as exc:
            try:  # Brevo says why, e.g. "sender is not valid" (never the key)
                why = str(json.loads(exc.read()).get("message", ""))[:200]
            except Exception:
                why = ""
            log.warning("mail to a user failed: Brevo answered %s %s", exc.code, why)
        except OSError as exc:
            log.warning("mail to a user failed: %s", type(exc).__name__)
        return False

    def _deliver(self, smtp: smtplib.SMTP, message: EmailMessage) -> None:
        if self.user and self.password:
            smtp.login(self.user, self.password)
        smtp.send_message(message)
