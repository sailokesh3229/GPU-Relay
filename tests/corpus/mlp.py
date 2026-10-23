"""Corpus 1: PyTorch MLP on synthetic data -- unmodified (no GPU Relay imports).

Records a ``losses`` list, prints per-epoch metrics, plots the losses, saves the model.
Made resumable automatically (loop instrumentation).
"""

import json
import os
import random
import time

import numpy as np
import torch

EPOCHS = int(os.environ.get("CORPUS_EPOCHS", "30"))
EPOCH_SLEEP = float(os.environ.get("CORPUS_EPOCH_SLEEP", "0"))

torch.set_num_threads(1)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

X = torch.randn(512, 16)
true_w = torch.randn(16, 1)
Y = X @ true_w + 0.1 * torch.from_numpy(np.random.randn(512, 1)).float()

model = torch.nn.Sequential(
    torch.nn.Linear(16, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(64, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
losses = []

print(f"training for {EPOCHS} epochs")
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X))
    total = 0.0
    for i in range(0, len(X), 64):
        idx = perm[i : i + 64]
        loss = torch.nn.functional.mse_loss(model(X[idx]), Y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    scheduler.step()
    jitter = random.random()  # exercises Python RNG continuity across resumes
    losses.append(total / len(X))
    lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch:3d} loss {losses[-1]:.6f} lr {lr:.5f} jitter {jitter:.4f}")
    if EPOCH_SLEEP:
        time.sleep(EPOCH_SLEEP)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    plt.figure()
    plt.plot(losses)
    plt.title("training loss")
    plt.show()

torch.save(model.state_dict(), "model.pt")
with open("metrics.json", "w") as fh:
    json.dump({"losses": losses}, fh)
print("done")
