"""Training loops inside functions (Phase 9 item 3, opt-in ``instrument_functions``).

The loop state lives in function locals (``history``, ``w``, ``rng``); with the option on,
safe points inside ``train`` snapshot and restore them. A ``while`` loop inside a function
is covered too when ``instrument_while`` is on.
"""

import os
import random
import time

STEP_SLEEP = float(os.environ.get("CORPUS_STEP_SLEEP", "0.1"))


def train(epochs, lr=0.05):
    rng = random.Random(3)
    history = []
    w = 2.0
    for epoch in range(epochs):
        grad = 2 * w + rng.gauss(0, 0.1)
        w -= lr * grad
        history.append(round(w, 9))
        print(f"epoch {epoch:2d} w {w:.9f}")
        time.sleep(STEP_SLEEP)
    return history


def refine(x):
    steps = 0
    while abs(x * x - 2) > 1e-12 and steps < 40:
        x = 0.5 * (x + 2 / x)
        steps += 1
        time.sleep(STEP_SLEEP / 2)
    return x, steps


def main():
    history = train(30)
    root, steps = refine(5.0)
    print(f"final w {history[-1]:.9f}, sqrt2 {root:.12f} after {steps} steps")


if __name__ == "__main__":
    main()
