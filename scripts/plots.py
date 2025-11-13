# confusion_matrix.py
import os
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from datasets_sequences import CustomDataset
from model_sequences import CNNGRU
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend for headless servers
import matplotlib.pyplot as plt

DATASET_PATH = "processed/dataset_sequences.parquet"
MODEL_PATH = "processed/model.pt"
IMAGE_SIZE = (256, 256)
SEQ_LEN = 5

# === CONFUSION MATRIX FUNCTION ===
def plot_confusion_matrices(model, loader, device, output_dir="processed/plots"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            preds = torch.sigmoid(model(seqs)).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    control_names = ["forward", "back", "left", "right"]

    for i, name in enumerate(control_names):
        y_true = (all_labels[:, i] > 0.5).astype(int)
        y_pred = (all_preds[:, i] > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"confusion_{name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
    print(f"✅ Confusion matrices saved to: {output_dir}")

# === Plot training curve ===
def plot_train_curve(history, output_dir="processed/plots"):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="Train Loss")
    plt.plot(history["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"train-val_loss.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Loss curve saved to {output_dir}")


# === Plot val curves ===
def plot_val_curve(history, output_dir="processed/plots"):
    plt.figure(figsize=(6, 4))
    plt.plot(history["val_exact"], label="Val exact")
    plt.plot([l[0] for l in history["val_labels_acc"]], label="acc frontward")
    plt.plot([l[1] for l in history["val_labels_acc"]], label="acc brack")
    plt.plot([l[2] for l in history["val_labels_acc"]], label="acc left")
    plt.plot([l[3] for l in history["val_labels_acc"]], label="acc right")
    plt.xlabel("Epoch")
    plt.ylabel("-")
    plt.title("Validation metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"val_evolution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Loss curve saved to {output_dir}")