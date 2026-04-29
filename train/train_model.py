"""
Model Training Script — MobileNetV2 Transfer Learning
------------------------------------------------------
Trains a binary crack classifier (cracked / non_cracked)
using the dataset in dataset/train and dataset/val.

USAGE:
  python train/train_model.py
  python train/train_model.py --epochs 20 --batch 32 --fine_tune
  python train/train_model.py --epochs 30 --batch 16 --fine_tune --fine_tune_layers 50

OUTPUT:
  models/crack_model.h5        ← saved model (used by the app)
  models/training_history.png  ← accuracy/loss plot
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Paths ───────────────────────────────────────────────────────────────────

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_SAVE_PATH = str(MODELS_DIR / "crack_model.h5")

# ─── Config ──────────────────────────────────────────────────────────────────

IMG_SIZE = (224, 224)
CLASSES = ["cracked", "non_cracked"]

# ─── Data Augmentation ───────────────────────────────────────────────────────

def get_data_generators(batch_size: int):
    """Create training and validation data generators with augmentation."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.75, 1.25],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(DATASET_DIR / "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASSES,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        str(DATASET_DIR / "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, val_gen


# ─── Model Building ──────────────────────────────────────────────────────────

def build_model(fine_tune: bool = False, fine_tune_layers: int = 30):
    """
    Build MobileNetV2-based binary classifier.
    Phase 1: Train only custom head (base frozen)
    Phase 2 (fine_tune=True): Unfreeze last N base layers
    """
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    if fine_tune:
        # Unfreeze the last `fine_tune_layers` layers of the base
        base.trainable = True
        for layer in base.layers[: -fine_tune_layers]:
            layer.trainable = False
        print(f"  🔓 Fine-tuning last {fine_tune_layers} layers of MobileNetV2")
    else:
        base.trainable = False
        print("  🔒 Base model frozen — training custom head only")

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=base.input, outputs=output)

    lr = 1e-5 if fine_tune else 1e-4
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ─── Training ────────────────────────────────────────────────────────────────

def train(epochs: int, batch_size: int, fine_tune: bool, fine_tune_layers: int):
    print("\n" + "=" * 55)
    print("  🏗️  Structural Damage Detector — Model Training")
    print("=" * 55)
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Fine-tune   : {fine_tune}")
    print(f"  GPU         : {'✅ ' + tf.test.gpu_device_name() if tf.config.list_physical_devices('GPU') else '❌ CPU only'}")
    print("=" * 55 + "\n")

    # Check dataset exists
    for split in ["train", "val"]:
        for cls in CLASSES:
            p = DATASET_DIR / split / cls
            if not p.exists() or len(list(p.iterdir())) == 0:
                print(f"❌ Missing or empty: {p}")
                print("   Run: python train/prepare_dataset.py --source <path>")
                return

    train_gen, val_gen = get_data_generators(batch_size)
    print(f"  Classes: {train_gen.class_indices}")
    print(f"  Train samples : {train_gen.samples}")
    print(f"  Val samples   : {val_gen.samples}\n")

    model = build_model(fine_tune=fine_tune, fine_tune_layers=fine_tune_layers)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir="models/logs", histogram_freq=0),
    ]

    start = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start

    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"   Model saved → {MODEL_SAVE_PATH}")

    plot_history(history)
    return history


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14)

    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plot_path = MODELS_DIR / "training_history.png"
    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=120)
    plt.close()
    print(f"   Plot saved  → {plot_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train structural damage detection model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16, reduce if OOM)")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Unfreeze last layers of MobileNetV2 for fine-tuning")
    parser.add_argument("--fine_tune_layers", type=int, default=30,
                        help="Number of base layers to unfreeze when fine-tuning (default: 30)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch,
        fine_tune=args.fine_tune,
        fine_tune_layers=args.fine_tune_layers,
    )


if __name__ == "__main__":
    main()
