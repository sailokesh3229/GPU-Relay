"""Corpus 6: Keras tiny model (Keras 3, torch backend) -- unmodified."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json
import time

import keras
import numpy as np

EPOCHS = int(os.environ.get("CORPUS_EPOCHS", "12"))
EPOCH_SLEEP = float(os.environ.get("CORPUS_EPOCH_SLEEP", "0"))

keras.utils.set_random_seed(0)
x = np.random.rand(512, 8).astype("float32")
y = (x @ np.linspace(-1, 1, 8)[:, None] + 0.05 * np.random.randn(512, 1)).astype("float32")

model = keras.Sequential(
    [
        keras.Input((8,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")


def report(epoch, logs):
    print(f"epoch {epoch:2d} loss {logs['loss']:.6f}")
    if EPOCH_SLEEP:
        time.sleep(EPOCH_SLEEP)


model.fit(
    x,
    y,
    epochs=EPOCHS,
    batch_size=32,
    verbose=0,
    shuffle=True,
    callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=report)],
)

final = float(model.evaluate(x, y, verbose=0))
with open("weights.json", "w") as fh:
    json.dump([w.tolist() for w in model.get_weights()], fh)
with open("metrics.json", "w") as fh:
    json.dump({"final_mse": final}, fh)
print(f"final mse {final:.6f}")
