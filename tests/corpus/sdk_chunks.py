"""Corpus 7 (SDK version): chunked evaluation over a list, no framework.

Each chunk's result is saved with ``relay.save_result`` so finished chunks survive
interruptions; the final metric is computed from ``relay.load_results()``.
"""

import json
import math
import os
import random
import time

import gpurelay as relay

N_ITEMS = int(os.environ.get("CORPUS_ITEMS", "400"))
CHUNK_SLEEP = float(os.environ.get("CORPUS_CHUNK_SLEEP", "0"))

random.seed(1234)
items = list(range(N_ITEMS))
seen = []
relay.state(seen=seen)


def expensive(x):
    return math.sin(x) * random.random()  # uses the global RNG on purpose


print(f"evaluating {N_ITEMS} items")
for chunk in relay.chunks(items, size=20):
    values = [expensive(x) for x in chunk.items]
    relay.save_result(chunk.id, values)
    seen.append(chunk.id)
    print(f"chunk {chunk.id:3d} mean {sum(values) / len(values):+.6f}")
    if CHUNK_SLEEP:
        time.sleep(CHUNK_SLEEP)

results = relay.load_results()
flat = [v for chunk_values in results for v in chunk_values]
summary = {"n": len(flat), "mean": sum(flat) / len(flat), "chunks_seen": seen}
with open("metrics.json", "w") as fh:
    json.dump(summary, fh)
print(f"mean {summary['mean']:.8f} over {summary['n']} items")
