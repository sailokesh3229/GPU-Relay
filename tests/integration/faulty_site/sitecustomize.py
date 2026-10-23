"""Test-only: make every storage a session opens flaky (``GPURELAY_TEST_FAULT_RATE``).

Put this directory on ``PYTHONPATH`` of a session subprocess; Python imports
``sitecustomize`` at start-up, before the runtime opens its storage.
"""

import os
import random

RATE = float(os.environ.get("GPURELAY_TEST_FAULT_RATE", "0") or 0)

if RATE:
    import gpurelay.storage.factory as _factory
    from gpurelay.errors import StorageError

    _rng = random.Random(os.getpid())
    _real_open = _factory.open_storage
    _OPS = {
        "put_file",
        "put_bytes",
        "get_file",
        "get_bytes",
        "list",
        "delete",
        "exists",
        "put_many",
        "delete_many",
        "sha256",
    }

    class FlakyStorage:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if name not in _OPS or not callable(attr):
                return attr

            def call(*args, **kwargs):
                if _rng.random() < RATE:
                    raise StorageError(f"injected fault in {name}")
                return attr(*args, **kwargs)

            return call

    def open_storage(uri):
        return FlakyStorage(_real_open(uri))

    _factory.open_storage = open_storage
