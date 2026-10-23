"""Adversarial user code under random kills (hardening campaign).

Each script is ordinary Python that stresses one assumption of the runtime. It runs once
uninterrupted and once through the relay with random hard kills; the merged log and output
files must be identical. A script that cannot be resumed must fail clearly, never produce
different results.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from equivalence import RunOutput, assert_equivalent, run_job

from gpurelay.config import RuntimeSettings

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETTINGS = RuntimeSettings(
    min_interval_s=0.3, max_lost_work_s=1.0, safety_margin_s=0.3, heartbeat_s=1,
    replay_threshold_s=0.5,
)  # fmt: skip

PRELUDE = "import time\nimport random\nSLEEP = 0.05\nrandom.seed(1)\n"

SCRIPTS = {
    # Python iterates a list by index while index < len(list): appends extend the loop.
    "growing_list": """
        items = [1]
        for x in items:
            time.sleep(SLEEP)
            if len(items) < 60:
                items.append(x + 1)
        print("visited", len(items), sum(items))
    """,
    # State that lives in a closure, created by a cheap (replayed) setup statement.
    "closure_counter": """
        def make_counter():
            count = [0]
            def bump(step):
                count[0] += step
                return count[0]
            return bump
        bump = make_counter()
        for i in range(60):
            time.sleep(SLEEP)
            value = bump(i)
        print("counter", value)
    """,
    # A user object changed only through its own methods (no assignment, no known mutator).
    "custom_object_methods": """
        class Tracker:
            def __init__(self):
                self.values = []
            def record(self, v):
                self.values.append(v)
            def summary(self):
                return len(self.values), round(sum(self.values), 9)
        tracker = Tracker()
        for i in range(60):
            time.sleep(SLEEP)
            tracker.record(random.random())
        print("tracker", tracker.summary())
    """,
    # break / continue / for-else, loop variable used after the loop.
    "control_flow": """
        found = []
        for i in range(80):
            time.sleep(SLEEP)
            if i % 7 == 3:
                continue
            found.append(random.random())
            if len(found) > 55:
                break
        else:
            found.append(-1.0)
        print("found", len(found), round(sum(found), 9), "last i", i)
    """,
    # Iterables that are not sequences: generator expression, enumerate, zip, dict items.
    "iterator_kinds": """
        weights = {f"k{j}": j * 0.5 for j in range(20)}
        total = 0.0
        for idx, (name, w) in enumerate(zip(weights, (v * v for v in weights.values()))):
            time.sleep(SLEEP * 2)
            total += idx * w + random.random()
        print("total", round(total, 9))
    """,
    # Global changed from inside a function called in the loop; attribute stores.
    "globals_and_attributes": """
        calls = 0
        class Box:
            pass
        box = Box()
        box.acc = 0.0
        def work(x):
            global calls
            calls += 1
            return x * 2
        for i in range(60):
            time.sleep(SLEEP)
            box.acc += work(i) + random.random()
        print("calls", calls, "acc", round(box.acc, 9))
    """,
    # Working files written in the loop and read back at the end.
    "working_files": """
        import os
        os.makedirs("parts", exist_ok=True)
        for i in range(50):
            time.sleep(SLEEP)
            with open(f"parts/{i:03d}.txt", "w") as fh:
                fh.write(str(random.random()))
        with open("summary.txt", "w") as fh:
            fh.write(str(sorted(os.listdir("parts"))[:3]) + str(len(os.listdir("parts"))))
        print("files", len(os.listdir("parts")))
    """,
    # A tight top-level loop (tiny iterations): sparse safe points must still resume it.
    "tight_loop": """
        walk = 0.0
        for i in range(25_000_000):
            walk += random.random() - 0.5
            if i % 5_000_000 == 0:
                print("at", i, round(walk, 9))
        print("walk", round(walk, 9))
    """,
    # Early exit from inside the loop with status 0: a completed job.
    "early_exit": """
        import sys
        for i in range(100):
            time.sleep(SLEEP)
            if i == 45:
                print("stopping at", i, round(random.random(), 9))
                sys.exit(0)
    """,
}


def _script(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(PRELUDE + textwrap.dedent(SCRIPTS[name]), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_adversarial_script_survives_kills(tmp_path: Path, name: str) -> None:
    script = _script(tmp_path, name)
    baseline = run_job(script, tmp_path / "a", settings=SETTINGS)
    assert baseline.outcomes == ["completed"]
    run: RunOutput = run_job(script, tmp_path / "b", settings=SETTINGS,
                             kill_between_s=(1.2, 2.0), seed=3)  # fmt: skip
    assert "crashed" in run.outcomes, run.outcomes
    assert_equivalent(baseline, run)
