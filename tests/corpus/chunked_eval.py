"""Corpus 7: chunked evaluation over a list, no framework -- unmodified script."""

import json
import math
import os
import random
import time

N_ITEMS = int(os.environ.get("CORPUS_ITEMS", "400"))
CHUNK_SLEEP = float(os.environ.get("CORPUS_CHUNK_SLEEP", "0"))
CHUNK = 20

random.seed(1234)
items = list(range(N_ITEMS))
results = []


def expensive(x):
    return math.sin(x) * random.random()  # uses the global RNG on purpose


print(f"evaluating {N_ITEMS} items")
for start in range(0, N_ITEMS, CHUNK):
    values = [expensive(x) for x in items[start : start + CHUNK]]
    results.append(values)
    print(f"chunk {start // CHUNK:3d} mean {sum(values) / len(values):+.6f}")
    if CHUNK_SLEEP:
        time.sleep(CHUNK_SLEEP)

flat = [v for chunk_values in results for v in chunk_values]
summary = {"n": len(flat), "mean": sum(flat) / len(flat), "chunks": len(results)}
with open("metrics.json", "w") as fh:
    json.dump(summary, fh)
print(f"mean {summary['mean']:.8f} over {summary['n']} items")
