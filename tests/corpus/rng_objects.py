"""Random generator *objects* created in cheap setup statements (replayed on resume) and
advanced inside the loop without being reassigned (Phase 9 follow-up: found on Kaggle GPU).

They must come back from the checkpoint, not from the replayed setup, or the random stream
restarts after a resume.
"""

import os
import random
import time

import numpy as np

SLEEP = float(os.environ.get("CORPUS_STEP_SLEEP", "0.2"))
rng = random.Random(42)
np_rng = np.random.default_rng(7)
legacy = np.random.RandomState(3)
total = 0.0
for step in range(30):
    time.sleep(SLEEP)
    total += rng.random() + float(np_rng.normal()) + float(legacy.uniform())
    print(f"step {step:2d} total {total:.9f}")
