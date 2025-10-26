#!/usr/bin/env python3
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.config import list_physical_devices
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import config as cfg
from dataset import get_datasets
from model import unet_model, bce_dice_loss, BinaryMeanIoU, DiceMetric


# --- Helpers to compute dataset sizes safely ---------------------------------
def count_image_files(folder):
    p = Path(folder)
    if not p.exists():
        return 0
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    count = 0
    for e in exts:
        count += len(list(p.glob(e)))
    return count


def dataset_cardinality(ds):
    try:
        c = tf.data.experimental.cardinality(ds).numpy()
        # cardinality returns -2 for unknown, -1 for infinite
        if c is None or c < 0:
            return None
        return int(c)
    except Exception:
        return None


# --- Training entrypoint ----------------------------------------------------
def main():
    # GPU check
    gpus = list_physical_devices("GPU")
    if gpus:
        print(f"✅ GPU is available: {gpus}")
    else:
        print("⚠️ No GPU detected, running on CPU.")

    # ---------------------------
    # Load datasets (function from dataset.py)
    # ---------------------------
    print("📂 Loading datasets...")
    train_dataset, val_dataset = get_datasets(val_split=0.2, batch_size=cfg.BATCH_SIZE)

    # ---------------------------
    # Build model
    # ---------------------------
    print("🧠 Building U-Net model...")
    model = unet_model(input_shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1))

    model.compile(
        optimizer=Adam(cfg.LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=["accuracy", BinaryMeanIoU(), DiceMetric()]
    )

    model.summary()

    # ---------------------------
    # Callbacks
    # ---------------------------
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=cfg.MODEL_PATH,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,  # a bit more patience for segmentation
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )

    callbacks = [checkpoint, early_stop, reduce_lr]

    # ---------------------------
    # Compute sensible steps_per_epoch / validation_steps if possible
    # ---------------------------
    # Try to get cardinality from the dataset first (works when dataset built without .repeat())
    train_card = dataset_cardinality(train_dataset)
    val_card = dataset_cardinality(val_dataset)

    # Fallback to counting files in folders (cfg should point to folders)
    if train_card is None:
        train_count = count_image_files(cfg.TRAIN_IMG_DIR)
        if train_count > 0:
            train_card = train_count // cfg.BATCH_SIZE or 1

    fit_kwargs = dict(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    if train_card is not None:
        fit_kwargs["steps_per_epoch"] = int(train_card)
    if val_card is not None:
        fit_kwargs["validation_steps"] = int(val_card)

    print("🚀 Starting training...")
    print(f"Training steps_per_epoch: {fit_kwargs.get('steps_per_epoch')}, "
          f"validation_steps: {fit_kwargs.get('validation_steps')}")

    history = model.fit(**fit_kwargs)

    # ---------------------------
    # Save training history
    # ---------------------------
    history_file = cfg.HISTORY_PATH
    with open(history_file, "w") as f:
        f.write(str(history.history))
    print(f"✅ Training complete. Best model saved to: {cfg.MODEL_PATH}")
    print(f"📊 Training history saved to: {history_file}")


if __name__ == "__main__":
    main()
