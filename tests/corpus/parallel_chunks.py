"""Independent chunks with relay.chunks(..., parallel=True) (Phase 9).

Each chunk draws random numbers and prints a line; results come back via save_result.
The relayed run (chunks spread over several backends) must match a sequential run.
"""

import os
import random
import time

import gpurelay as relay

N = int(os.environ.get("CORPUS_ITEMS", "320"))
SIZE = int(os.environ.get("CORPUS_CHUNK", "20"))
SLEEP = float(os.environ.get("CORPUS_CHUNK_SLEEP", "0.8"))

random.seed(7)
offset = random.random()
print(f"evaluating {N} items in chunks of {SIZE}")

for chunk in relay.chunks(range(N), SIZE, parallel=True, seed=11):
    time.sleep(SLEEP)
    values = [x * offset + random.random() for x in chunk]
    print(f"chunk {chunk.id:3d}: mean {sum(values) / len(values):.6f}")
    relay.save_result(chunk.id, values)

results = relay.load_results()
flat = [v for part in results for v in part]
print(f"total {sum(flat):.6f} from {len(results)} chunks")
print(f"after loop: {random.random():.6f}")
with open("summary.txt", "w") as fh:
    fh.write(f"{sum(flat):.12f}\n")
