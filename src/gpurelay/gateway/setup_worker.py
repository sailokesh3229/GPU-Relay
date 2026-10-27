"""One setup check for one user, in its own process (plan M2).

Run by the gateway as ``python -m gpurelay.gateway.setup_worker`` with the environment from
:func:`gpurelay.gateway.userenv.user_environment` (only that user's tokens) and the user's
``config.toml`` in ``$GPURELAY_HOME``. Reads one JSON request on stdin, updates the config
file on success, and prints one JSON line: ``{"ok": true, ...}`` or ``{"error": "..."}``.

Steps:

* ``hf``: the token is accepted; the private storage repo is created or verified.
* ``kaggle``: the Kaggle token works; the private secrets dataset that lets Kaggle sessions
  reach the storage repo is created (or refreshed with ``rotate``).
* ``modal``: the Modal token works (the SDK's ``hello`` call); Modal is switched on.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from gpurelay.config import load_config, save_config
from gpurelay.errors import GpuRelayError


def _hf(request: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    from gpurelay.controller.setup import setup_storage

    token = os.environ.get("HF_TOKEN", "")
    if not token.startswith("hf_"):
        raise GpuRelayError("a Hugging Face token starts with hf_")
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception as exc:
        raise GpuRelayError(f"Hugging Face did not accept this token: {exc}") from exc
    config = load_config()
    try:
        result = setup_storage(config, api, repo=(request.get("repo") or "").strip() or None)
    except GpuRelayError:
        raise
    except Exception as exc:
        raise GpuRelayError(f"could not set up the storage repo (does the token have write "
                            f"access?): {exc}") from exc  # fmt: skip
    save_config(config)
    return {"user": result.hf_user, "repo": result.repo_id, "created": result.created}


def _kaggle(request: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    from gpurelay.backends.kaggle import OfficialKaggleClient
    from gpurelay.controller.setup import setup_kaggle, validate_session_token

    config = load_config()
    repo = config.storage.hf_repo_id
    if not repo:
        raise GpuRelayError("connect Hugging Face first")
    session = os.environ.get("HF_TOKEN", "")
    validate_session_token(session, repo, HfApi())
    try:
        client = OfficialKaggleClient()
        result = setup_kaggle(config, client, session_token=session, repo_id=repo,
                              rotate=bool(request.get("rotate")))  # fmt: skip
    except GpuRelayError:
        raise
    except Exception as exc:
        raise GpuRelayError(f"Kaggle did not accept this token: {exc}") from exc
    accelerator = request.get("accelerator")
    if accelerator:
        config.backends.kaggle.accelerator = str(accelerator)
    save_config(config)
    return {"username": result.username, "created": result.created}


def _modal(request: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    if not config.storage.hf_repo_id:
        raise GpuRelayError("connect Hugging Face first")
    if not request.get("terms"):
        raise GpuRelayError("confirm that you read Modal's terms of service")
    try:
        import modal

        client = modal.Client.from_credentials(
            os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"]
        )
        client.hello()
    except ImportError as exc:
        raise GpuRelayError("the server has no Modal SDK (pip install 'gpurelay[modal]')") from exc
    except Exception as exc:
        raise GpuRelayError(f"Modal did not accept this token: {exc}") from exc
    modal_settings = config.backends.modal
    modal_settings.terms_reviewed = True  # first: enabling Modal is validated against it
    modal_settings.enabled = True
    modal_settings.gpu = str(request.get("gpu") or modal_settings.gpu)
    save_config(config)
    return {"gpu": modal_settings.gpu}


STEPS = {"hf": _hf, "kaggle": _kaggle, "modal": _modal}


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        step = STEPS.get(str(request.get("step")))
        if step is None:
            raise GpuRelayError("unknown setup step")
        result: dict[str, Any] = {"ok": True, **step(request)}
    except GpuRelayError as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # never a traceback (it could quote a token) to the page
        result = {"error": f"setup failed ({type(exc).__name__})"}
    sys.stdout.write("\n" + json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
