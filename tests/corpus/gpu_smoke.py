"""Kaggle GPU smoke test (Phase 9 follow-up): a short CUDA job that hands off.

Prints the device, an integer checksum computed on the GPU (exact on any device), and a
small training loss rounded to 3 decimals, so the relayed GPU log can be compared with a
CPU run of the same script.
"""

import os
import time

import torch

STEPS = int(os.environ.get("SMOKE_STEPS", "60"))  # ~10 min: at least one handoff
SLEEP = float(os.environ.get("SMOKE_SLEEP", "10"))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", torch.cuda.get_device_name(0) if device == "cuda" else "cpu", flush=True)

torch.manual_seed(0)
X = torch.randn(512, 16)
y = X @ torch.linspace(-1, 1, 16) + 0.1 * torch.randn(512)
model = torch.nn.Linear(16, 1).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
gen = torch.Generator().manual_seed(1)
checksum = 0

for step in range(STEPS):
    ints = torch.randint(0, 1000, (256, 256), generator=gen).to(device)
    checksum = (checksum * 31 + int((ints * ints.flip(0)).sum().item())) % (2**61 - 1)
    loss = torch.nn.functional.mse_loss(model(X.to(device)).squeeze(1), y.to(device))
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {step:2d} checksum {checksum} loss {loss.item():.3f}", flush=True)
    time.sleep(SLEEP)

print(f"done on {device}")
