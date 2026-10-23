"""Mid-epoch resume of a shuffled DataLoader (Phase 9): epochs are longer than sessions.

Sessions train for less than an epoch (~6 s) before they die, so the job only finishes if
it resumes inside an epoch, with the same shuffled batch order (checked by the weights).
"""

import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X = torch.randn(2048, 8)
y = (X.sum(1) > 0).float()
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)  # 128 batches
model = torch.nn.Linear(8, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
SLEEP = float(os.environ.get("CORPUS_STEP_SLEEP", "0.05"))

for epoch in range(3):
    total = 0.0
    for xb, yb in loader:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb).squeeze(1), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        time.sleep(SLEEP)
    print(f"epoch {epoch} loss {total / len(loader):.6f}")
print("weights", [round(v, 6) for v in model.weight.flatten().tolist()])
