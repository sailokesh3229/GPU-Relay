"""Corpus 5: PyTorch Lightning tiny model -- unmodified."""

import json
import logging
import os
import time

import lightning.pytorch as L
import torch

EPOCHS = int(os.environ.get("CORPUS_EPOCHS", "10"))
EPOCH_SLEEP = float(os.environ.get("CORPUS_EPOCH_SLEEP", "0"))

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
torch.set_num_threads(1)
L.seed_everything(0, verbose=False)

x = torch.randn(512, 8)
y = x @ torch.linspace(-1, 1, 8).unsqueeze(1) + 0.1 * torch.randn(512, 1)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True
)


class Regressor(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(32, 1)
        )
        self.epoch_losses = []

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        loss = torch.nn.functional.mse_loss(self.net(xb), yb)
        self.epoch_losses.append(loss.item())
        return loss

    def on_train_epoch_start(self):
        self.epoch_losses = []

    def on_train_epoch_end(self):
        mean = sum(self.epoch_losses) / len(self.epoch_losses)
        print(f"epoch {self.current_epoch:2d} loss {mean:.6f}")
        if EPOCH_SLEEP:
            time.sleep(EPOCH_SLEEP)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


model = Regressor()
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="cpu",
    devices=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    enable_checkpointing=False,
    logger=False,
    deterministic=True,
)
trainer.fit(model, loader)

torch.save(model.state_dict(), "model.pt")
with torch.no_grad():
    final = torch.nn.functional.mse_loss(model.net.eval()(x), y).item()
with open("metrics.json", "w") as fh:
    json.dump({"final_mse": final, "epochs": trainer.current_epoch}, fh)
print(f"final mse {final:.6f}")
