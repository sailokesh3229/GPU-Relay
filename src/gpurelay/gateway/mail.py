"""Email from the website (plan M4): verification, password reset, job notices.

Optional. With ``GPURELAY_SMTP_HOST`` set the site sends mail through that server (any
provider with SMTP: Gmail with an app password, Brevo, Mailgun, ...); new accounts then
verify their email before they connect platforms, and "forgot password" works. Without it
there is no email at all: accounts need no verification and a forgotten password cannot be
reset (the people you invite should keep theirs).

* ``GPURELAY_SMTP_HOST``, ``GPURELAY_SMTP_PORT`` (587: STARTTLS; 465: TLS);
* ``GPURELAY_SMTP_USER``, ``GPURELAY_SMTP_PASSWORD``;
* ``GPURELAY_MAIL_FROM`` (default: the SMTP user);
* ``GPURELAY_SITE_URL``: the site's address for links in mails (``https://...``).
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
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


@dataclass
class Mailer:
    host: str
    port: int
    user: str | None
    password: str | None
    sender: str
    site_url: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Mailer | None:
        env = os.environ if env is None else env
        host = env.get(ENV_HOST, "").strip()
        if not host:
            return None
        site = env.get(ENV_SITE_URL, "").strip().rstrip("/")
        if not site.startswith(("https://", "http://")):
            raise ConfigError(f"set {ENV_SITE_URL} (https://...) so mails can link to the site")
        user = env.get(ENV_USER, "").strip() or None
        sender = env.get(ENV_FROM, "").strip() or user
        if not sender:
            raise ConfigError(f"set {ENV_FROM}: the address mails come from")
        try:
            port = int(env.get(ENV_PORT, "") or 587)
        except ValueError:
            raise ConfigError(f"{ENV_PORT} must be a number") from None
        return cls(host, port, user, env.get(ENV_PASSWORD) or None, sender, site)

    def link(self, path: str) -> str:
        return self.site_url + path

    def send(self, to: str, subject: str, body: str) -> bool:
        """Send one plain-text mail; False (logged, never raised) if the server refused it."""
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

    def _deliver(self, smtp: smtplib.SMTP, message: EmailMessage) -> None:
        if self.user and self.password:
            smtp.login(self.user, self.password)
        smtp.send_message(message)
