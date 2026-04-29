"""
Model Evaluation Script
------------------------
Evaluates the trained model on the test set and generates:
  - Classification report (precision, recall, F1)
  - Confusion matrix plot
  - ROC curve plot
  - Sample predictions grid

USAGE:
  python train/evaluate_model.py
  python train/evaluate_model.py --samples 20
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Config ──────────────────────────────────────────────────────────────────

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "crack_model.h5"
IMG_SIZE = (224, 224)
CLASSES = ["cracked", "non_cracked"]
CLASS_LABELS = {0: "Cracked", 1: "Non-Cracked"}


def load_model():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Run: python train/train_model.py")
        exit(1)
    print(f"✅ Loading model from {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH))


def get_test_generator(batch_size=16):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        str(DATASET_DIR / "test"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )


def evaluate(model, test_gen):
    print("\n🔍 Running evaluation on test set...")
    results = model.evaluate(test_gen, verbose=1)
    metric_names = model.metrics_names
    print("\n📊 Test Metrics:")
    for name, val in zip(metric_names, results):
        print(f"   {name:15} : {val:.4f}")
    return results


def generate_predictions(model, test_gen):
    """Get all predictions and true labels."""
    y_true = test_gen.classes
    y_pred_raw = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_raw > 0.5).astype(int).flatten()
    return y_true, y_pred, y_pred_raw.flatten()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cracked", "Non-Cracked"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(str(path), dpi=120)
    plt.close()
    print(f"   Saved → {path}")


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = MODELS_DIR / "roc_curve.png"
    plt.savefig(str(path), dpi=120)
    plt.close()
    print(f"   Saved → {path}")


def plot_sample_predictions(model, test_gen, n_samples=16):
    """Plot a grid of test images with predicted vs true labels."""
    test_dir = DATASET_DIR / "test"
    all_images = []
    for cls in CLASSES:
        folder = test_dir / cls
        if folder.exists():
            imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            for img_path in random.sample(imgs, min(n_samples // 2, len(imgs))):
                all_images.append((img_path, cls))

    random.shuffle(all_images)
    n = min(n_samples, len(all_images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Sample Predictions", fontsize=16)

    datagen_norm = lambda img: img.astype(np.float32) / 255.0

    for i, (img_path, true_cls) in enumerate(all_images[:n]):
        import cv2
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        inp = np.expand_dims(datagen_norm(img_resized), 0)
        score = float(model.predict(inp, verbose=0)[0][0])
        pred_cls = "cracked" if score > 0.5 else "non_cracked"
        correct = pred_cls == true_cls

        axes[i].imshow(img_resized)
        axes[i].set_title(
            f"True: {true_cls}\nPred: {pred_cls} ({score:.0%})",
            fontsize=8,
            color="green" if correct else "red",
        )
        axes[i].axis("off")
        for spine in axes[i].spines.values():
            spine.set_edgecolor("green" if correct else "red")
            spine.set_linewidth(2)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = MODELS_DIR / "sample_predictions.png"
    plt.savefig(str(path), dpi=100)
    plt.close()
    print(f"   Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--samples", type=int, default=16,
                        help="Number of sample predictions to display (default: 16)")
    args = parser.parse_args()

    model = load_model()
    test_gen = get_test_generator()

    if test_gen.samples == 0:
        print("❌ No test images found in dataset/test/")
        return

    evaluate(model, test_gen)

    print("\n📈 Generating evaluation plots...")
    y_true, y_pred, y_scores = generate_predictions(model, test_gen)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Cracked", "Non-Cracked"]))

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_scores)
    plot_sample_predictions(model, test_gen, n_samples=args.samples)

    print("\n✅ Evaluation complete! Check the models/ folder for plots.")


if __name__ == "__main__":
    main()
