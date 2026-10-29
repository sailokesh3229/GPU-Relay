"""A function's result kept by its caller (opt-in ``instrument_functions``).

``history`` comes from ``train()``, whose loop ends before every checkpoint (they are taken in
``evaluate()``'s loop, which is the slow one). A resume skips ``train()``'s loop, so ``main``'s
``history`` must come back from the checkpoint.
"""

import os
import random
import time

STEP_SLEEP = float(os.environ.get("CORPUS_STEP_SLEEP", "0.1"))


def train(epochs):
    rng = random.Random(5)
    history = []
    w = 1.0
    for _epoch in range(epochs):
        w -= 0.1 * (2 * w + rng.gauss(0, 0.1))
        history.append(round(w, 9))
    return history


def evaluate(history, rounds):
    scores = []
    for r in range(rounds):
        scores.append(round(sum(history) * (r + 1), 9))
        print(f"round {r:2d} score {scores[-1]:.9f}")
        time.sleep(STEP_SLEEP)
    return scores


def main():
    history = train(20)
    scores = evaluate(history, 40)
    print(f"final w {history[-1]:.9f}, score {scores[-1]:.9f}")


if __name__ == "__main__":
    main()
