"""One user's GPU Relay on the server (plan M3): their web app and controller.

Started by the gateway as ``python -m gpurelay.gateway.user_server --port N`` with the
environment from :func:`gpurelay.gateway.userenv.user_environment` (only this user's tokens)
plus ``GPURELAY_LAUNCH_KEY``. It listens on 127.0.0.1 only, and every request needs the
launch key (the gateway adds it), so only the gateway reaches it.

On start it restores the user's job database from their own storage repo (the server's disk
is temporary), keeps the settings the gateway wrote (the gateway's database is where
settings live), then runs the controller under the usual supervisor, which backs the job
database up to the user's repo. ``server.stop`` in the home stops it gracefully.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import threading

from gpurelay.config import gpurelay_home, load_config, save_config
from gpurelay.constants import CONFIG_FILENAME

ENV_LAUNCH_KEY = "GPURELAY_LAUNCH_KEY"
STOP_NAME = "server.stop"

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - wires real pieces
    import uvicorn

    from gpurelay.controller.service import open_context, open_storage_from_config
    from gpurelay.dashboard.app import LOCAL_HOSTS, create_app
    from gpurelay.gateway.userenv import server_rules
    from gpurelay.hosted import (
        StateBackup,
        Supervisor,
        controller_command,
        restore_state,
        watch_stop_file,
    )

    parser = argparse.ArgumentParser(prog="gpurelay.gateway.user_server")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [user] %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "hf_xet", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    key = os.environ.pop(ENV_LAUNCH_KEY)  # not passed on to the controller
    home = gpurelay_home()
    config_path = home / CONFIG_FILENAME
    config = load_config(config_path)  # the settings the gateway wrote
    storage = open_storage_from_config(config, home)
    if restore_state(storage, home):
        log.info("restored the job database from the storage repo")
    save_config(server_rules(config), config_path)  # a restored backup never wins
    supervisor = Supervisor(controller_command(takeover=True), StateBackup(storage, home),
                            home=home)  # fmt: skip
    threading.Thread(target=supervisor.run, name="gpurelay-supervisor", daemon=True).start()
    watch_stop_file(home, name=STOP_NAME)
    try:
        ctx = open_context(load_config(config_path))
        uvicorn.run(create_app(ctx, allowed_hosts=LOCAL_HOSTS, launch_key=key),
                    host="127.0.0.1", port=args.port, log_level="warning")  # fmt: skip
    finally:
        with contextlib.suppress(Exception):
            supervisor.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
