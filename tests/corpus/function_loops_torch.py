"""A torch training loop inside a function (Phase 9 item 3, opt-in): the model and optimizer
are function locals; on resume their state is loaded into the live (re-created) objects."""

import os
import time

import torch


def train(epochs):
    torch.manual_seed(0)
    X = torch.randn(256, 4)
    y = X @ torch.tensor([1.0, -2.0, 0.5, 3.0]) + 0.1 * torch.randn(256)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        loss = torch.nn.functional.mse_loss(model(X[perm]).squeeze(1), y[perm])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 6))
        print(f"epoch {epoch:2d} loss {loss.item():.6f}")
        time.sleep(float(os.environ.get("CORPUS_STEP_SLEEP", "0.1")))
    return model, losses


model, losses = train(40)
print("weights", [round(v, 5) for v in model.weight.flatten().tolist()])
