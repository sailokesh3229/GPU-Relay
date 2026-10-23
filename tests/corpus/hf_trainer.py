"""Corpus 4: Hugging Face Trainer with a tiny randomly initialised model -- unmodified."""

import json
import os
import time

import torch
import transformers
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

EPOCHS = int(os.environ.get("CORPUS_EPOCHS", "4"))
STEP_SLEEP = float(os.environ.get("CORPUS_STEP_SLEEP", "0"))

transformers.logging.set_verbosity_error()
torch.set_num_threads(1)
set_seed(0)


class ToyDataset(torch.utils.data.Dataset):
    """Token sequences labelled by the parity of their sum."""

    def __init__(self, n=256, length=12, vocab=64):
        gen = torch.Generator().manual_seed(7)
        self.ids = torch.randint(1, vocab, (n, length), generator=gen)
        self.labels = (self.ids.sum(dim=1) % 2).long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i]}


config = BertConfig(
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=32,
    num_labels=2,
)
model = BertForSequenceClassification(config)


class SlowDown(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if STEP_SLEEP:
            time.sleep(STEP_SLEEP)


class PrintLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"step {state.global_step:3d} loss {logs['loss']:.6f}")


args = TrainingArguments(
    output_dir="hf_out",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16,
    learning_rate=5e-3,
    logging_steps=4,
    save_strategy="no",
    report_to="none",
    disable_tqdm=True,
    seed=0,
    use_cpu=True,
    dataloader_num_workers=0,
)
trainer = Trainer(
    model=model, args=args, train_dataset=ToyDataset(), callbacks=[SlowDown(), PrintLoss()]
)
trainer.remove_callback(PrinterCallback)  # its log dicts include wall-clock timings
trainer.train()

losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
torch.save(model.state_dict(), "model.pt")
with open("metrics.json", "w") as fh:
    json.dump({"losses": losses, "global_step": trainer.state.global_step}, fh)
print(f"trained {trainer.state.global_step} steps")
