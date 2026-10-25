"""Relay demo: CUDA training mixing the resume cases the hardening tests found.

Every printed value is exact on any device (integer checksums, losses rounded to 3
decimals), so the relayed GPU log can be compared with a local CPU run.
"""

import os
import random
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

SLEEP = float(os.environ.get("STRESS_SLEEP", "10"))
EPOCHS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", torch.cuda.get_device_name(0) if device == "cuda" else "cpu", flush=True)

torch.manual_seed(0)
X = torch.randn(480, 16)
y = X @ torch.linspace(-1, 1, 16) + 0.1 * torch.randn(480)
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)  # 30 batches/epoch
model = torch.nn.Linear(16, 1).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
gen = torch.Generator().manual_seed(1)
rng = random.Random(5)


class Stats:
    def __init__(self):
        self.seen = 0
        self.total = 0.0

    def record(self, value):
        self.seen += 1
        self.total += value


def make_counter():
    count = 0

    def bump():
        nonlocal count
        count += 1
        return count

    return bump


stats = Stats()
bump = make_counter()
checksum = 0
queue = [1, 2, 3]
for item in queue:  # grows while iterated
    if item < 40:
        queue.append(item + 3)

for epoch in range(EPOCHS):
    for step, (xb, yb) in enumerate(loader):
        ints = torch.randint(0, 1000, (128, 128), generator=gen).to(device)
        checksum = (checksum * 31 + int((ints * ints.flip(0)).sum().item())) % (2**61 - 1)
        loss = torch.nn.functional.mse_loss(model(xb.to(device)).squeeze(1), yb.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        stats.record(round(loss.item(), 3))
        n = bump()
        pick = rng.randint(0, 999)
        batch_sum = int((xb * 1000).round().sum().item())
        line = (
            f"e{epoch} s{step:2d} n{n:3d} cs {checksum} batch {batch_sum} pick {pick} "
            f"loss {loss.item():.3f}"
        )
        print(line, flush=True)
        with open("steps.txt", "a") as fh:
            fh.write(line + "\n")
        time.sleep(SLEEP)

print(f"queue {len(queue)} sum {sum(queue)}")
print(f"stats seen {stats.seen} mean {stats.total / stats.seen:.3f} counter {bump()}")
print(f"done on {device}")
