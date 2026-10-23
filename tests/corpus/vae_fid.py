"""Corpus 3: tiny VAE + FID-style metric computed in chunks -- unmodified script.

The owner's original use case, miniaturized: train a VAE, then compute a Frechet
distance between features of real and generated images, using a small fixed CNN as the
feature extractor (an "Inception" stand-in), in chunks.
"""

import json
import os
import random
import time

import numpy as np
import torch
from torch import nn

EPOCHS = int(os.environ.get("CORPUS_EPOCHS", "12"))
EPOCH_SLEEP = float(os.environ.get("CORPUS_EPOCH_SLEEP", "0"))
N_GEN = int(os.environ.get("CORPUS_N_GEN", "512"))
CHUNK = 64
LATENT = 4

torch.set_num_threads(1)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def make_images(n):
    """Synthetic 8x8 'digits': one Gaussian blob per image."""
    gen = torch.Generator().manual_seed(123)
    centers = torch.randint(1, 7, (n, 2), generator=gen).float()
    yy, xx = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    d2 = (yy[None] - centers[:, 0, None, None]) ** 2 + (xx[None] - centers[:, 1, None, None]) ** 2
    return torch.exp(-d2 / 3.0).unsqueeze(1)


data = make_images(1024)


class VAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2 * latent)
        )
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())

    def decode(self, z):
        return self.dec(z).view(-1, 1, 8, 8)

    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


vae = VAE()
optimizer = torch.optim.Adam(vae.parameters(), lr=3e-3)
history = []

print(f"training VAE for {EPOCHS} epochs")
for epoch in range(EPOCHS):
    perm = torch.randperm(len(data))
    total = 0.0
    for i in range(0, len(data), 128):
        x = data[perm[i : i + 128]]
        recon, mu, logvar = vae(x)
        rec = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (rec + kld) / len(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    history.append(total / len(data))
    print(f"epoch {epoch:2d} loss {history[-1]:.5f}")
    if EPOCH_SLEEP:
        time.sleep(EPOCH_SLEEP)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    plt.figure()
    plt.plot(history)
    plt.title("VAE loss")
    plt.show()
torch.save(vae.state_dict(), "vae.pt")

# Fixed feature extractor (stand-in for Inception).
torch.manual_seed(1234)
extractor = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten()
).eval()
for p in extractor.parameters():
    p.requires_grad_(False)

real_feats = []
with torch.no_grad():
    for start in range(0, len(data), CHUNK):
        real_feats.append(extractor(data[start : start + CHUNK]).numpy())

vae.eval()
gen_feats = []
with torch.no_grad():
    for start in range(0, N_GEN, CHUNK):
        z = torch.randn(CHUNK, LATENT)
        gen_feats.append(extractor(vae.decode(z)).numpy())
        print(f"generated {start + CHUNK}/{N_GEN}")
        if EPOCH_SLEEP:
            time.sleep(EPOCH_SLEEP)


def frechet_distance(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean_trace = np.sqrt(np.clip(np.linalg.eigvals(s1 @ s2).real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * covmean_trace)


score = frechet_distance(np.concatenate(real_feats), np.concatenate(gen_feats))
print(f"FID-style score: {score:.6f}")
with open("metrics.json", "w") as fh:
    json.dump({"history": history, "fid": score}, fh)
