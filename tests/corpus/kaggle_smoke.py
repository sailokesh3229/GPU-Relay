"""Smoke-test job for real backends (docs/manual-tests.md) -- unmodified, no GPU needed.

Deterministic on every machine (integer arithmetic + the ``random`` module only), so a
relayed Kaggle run can be compared byte-for-byte with a local uninterrupted run. Slow by
default (~20 min) so that a ~10 min session limit forces real handoffs.

    SMOKE_STEPS (default 40)   SMOKE_SLEEP seconds per step (default 30)
"""

import hashlib
import json
import os
import random
import time

STEPS = int(os.environ.get("SMOKE_STEPS", "40"))
SLEEP = float(os.environ.get("SMOKE_SLEEP", "30"))

random.seed(2026)
history = []
state = 1
print(f"smoke test: {STEPS} steps")
for step in range(STEPS):
    for _ in range(20_000):
        state = (state * 6364136223846793005 + random.getrandbits(32)) % (2**61 - 1)
    digest = hashlib.sha256(str(state).encode()).hexdigest()[:12]
    history.append(digest)
    print(f"step {step:3d} state {digest}")
    time.sleep(SLEEP)

with open("result.json", "w") as fh:
    json.dump({"steps": STEPS, "final": state, "history": history}, fh)
print(f"done: final state {hashlib.sha256(str(state).encode()).hexdigest()}")
