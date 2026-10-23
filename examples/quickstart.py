"""GPU Relay quick-start example: a tiny "training" loop with no dependencies.

Run it under GPU Relay and interrupt it as often as you like: the final output is identical
to an uninterrupted run (state, random numbers, and printed output are all restored).
"""

import json
import random
import time

random.seed(0)
STEPS = 40
weights = [random.uniform(-1, 1) for _ in range(8)]
target = [0.5] * 8
history = []

for step in range(STEPS):
    time.sleep(0.5)  # stands in for real GPU work
    grads = [2 * (w - t) + random.gauss(0, 0.05) for w, t in zip(weights, target, strict=True)]
    weights = [w - 0.1 * g for w, g in zip(weights, grads, strict=True)]
    loss = sum((w - t) ** 2 for w, t in zip(weights, target, strict=True))
    history.append(loss)
    print(f"step {step:2d}  loss {loss:.6f}")

with open("result.json", "w") as fh:
    json.dump({"weights": weights, "history": history}, fh, indent=2)
print("done")
