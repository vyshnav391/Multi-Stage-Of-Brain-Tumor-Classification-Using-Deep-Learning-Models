"""
evaluator.py
------------
Evaluation utilities: confusion matrix, classification report,
per-class accuracy, and training curve plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_curves(h1, h2, save_dir: str = "results/plots"):
    """Plot accuracy, loss and AUC curves across both training phases."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    for metric, title in [
        ("accuracy", "Model Accuracy"),
        ("loss",     "Model Loss"),
        ("auc",      "Model AUC"),
    ]:
        if metric not in h1.history:
            continue

        full     = h1.history[metric]     + h2.history.get(metric, [])
        full_val = h1.history[f"val_{metric}"] + h2.history.get(f"val_{metric}", [])
        split    = len(h1.history[metric]) - 1

        plt.figure(figsize=(10, 4))
        plt.plot(full, label="Train")
        plt.plot(full_val, label="Validation")
        plt.axvline(x=split, color="gray", linestyle="--", label="Fine-tune start")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = f"{save_dir}/{metric}_curve.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"  Saved: {path}")


def evaluate_model(model, val_data, class_names: list, save_dir: str = "results"):
    """
    Full evaluation: prints report, plots confusion matrix.

    Args:
        model:       Trained Keras model.
        val_data:    Validation DirectoryIterator.
        class_names: List of class label strings.
        save_dir:    Directory to save output figures.

    Returns:
        (y_true, y_pred): Ground-truth and predicted label arrays.
    """
    import os
    os.makedirs(f"{save_dir}/plots", exist_ok=True)

    val_data.reset()
    y_pred_probs = model.predict(val_data, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_data.classes

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n📋 Classification Report:\n", report)
    with open(f"{save_dir}/reports/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = f"{save_dir}/plots/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"  Saved: {cm_path}")

    # Per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    for i, cls in enumerate(class_names):
        mask = y_true == i
        acc = np.sum(y_pred[mask] == i) / np.sum(mask) if np.sum(mask) > 0 else 0
        print(f"  {cls:20s}: {acc * 100:.1f}%")

    return y_true, y_pred
