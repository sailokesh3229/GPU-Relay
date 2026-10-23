"""Corpus 8: main work in a ``while`` loop -- not instrumented in v1 (expected PARTIAL).

The ``while`` block restarts from its beginning after an interruption; the result must
still be correct.
"""

import json
import os
import random
import time

STEP_SLEEP = float(os.environ.get("CORPUS_STEP_SLEEP", "0"))

random.seed(7)
samples = []
for _ in range(20):
    samples.append(random.random())
    if STEP_SLEEP:
        time.sleep(STEP_SLEEP)
print(f"sampled {len(samples)} values, mean {sum(samples) / len(samples):.6f}")

x = 1.0
steps = 0
while steps < 30:
    x = 0.5 * (x + 2 / x) + random.random() * 1e-3 / (steps + 1) ** 2
    steps += 1
    print(f"step {steps:2d} x {x:.12f}")
    if STEP_SLEEP:
        time.sleep(STEP_SLEEP)

with open("metrics.json", "w") as fh:
    json.dump({"x": x, "steps": steps, "samples": samples}, fh)
print("done")
