"""Bindings made where replay does not recreate them: imports inside try/if blocks.

They follow a slow statement, so a resume skips them (replay stops at the first expensive
statement): ``js`` and ``OD`` must come back from the checkpoint (imported again by name),
or the loop fails with a NameError.
"""

import os
import time

STEP_SLEEP = float(os.environ.get("CORPUS_STEP_SLEEP", "0.3"))
time.sleep(4 * STEP_SLEEP)  # slow setup (like a training loop): skipped on resume, ends replay

try:
    import json as js
except ImportError:
    js = None
if os.environ.get("CORPUS_FLAG", "1") == "1":
    from collections import OrderedDict as OD
else:
    OD = dict

totals = OD()
for i in range(14):
    time.sleep(STEP_SLEEP)
    totals[f"k{i}"] = i * i
    print(js.dumps({"step": i, "sum": sum(totals.values())}))
print("done", js.dumps(totals))
