# train_sequences.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

from datasets_sequences import CustomDataset
from model_sequences import CNNGRU
from plots import plot_confusion_matrices, plot_train_curve, plot_val_curve

from pathlib import Path
import time
import os

# === CONFIG ===
DATASET_PATH = "processed/dataset_sequences.parquet"
IMAGE_SIZE   = (256, 256)
SEQ_LEN      = 5
MAX_SPEED    = 50

BATCH_SIZE   = 8
NUM_WORKER   = 2
TRAIN_SPLIT  = 0.8

EPOCHS       = 100
PATIENCE     = 10
MIN_DELTA    = 1e-2

LR           = 1e-4
WEIGHT_DECAY = 1e-6
MODEL_SAVE_PATH = "processed/model.pt"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE.type == "cuda" else False

# === LOSS FUNCTION ===
class MixedLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss for binary control signals and MSE for continuous speed.
    BCEWithLogitsLoss expects raw logits (no sigmoid).
    """
    def __init__(self, speed_weight=0.2):
        super().__init__()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.speed_weight = speed_weight

    def forward(self, preds, targets):
        # preds, targets: (B, 5)
        # controls: first 4 entries (raw logits)
        controls_pred_logits = preds[:, :4]
        controls_true = targets[:, :4]
        speed_pred = preds[:, 4]
        speed_true = targets[:, 4] / MAX_SPEED

        loss_controls = self.bce_logits(controls_pred_logits, controls_true)
        loss_speed = self.mse(speed_pred, speed_true)
        return loss_controls + self.speed_weight * loss_speed

# === TRAIN LOOP ===
def train():
    # ============================================================
    # LOAD DATASET
    # ============================================================
    print("Loading dataset...")
    dataset = CustomDataset(DATASET_PATH)
    n = len(dataset)
    n_train = int(n * TRAIN_SPLIT)
    n_val = n - n_train
    print(f"Total sequences: {n}")
    print(f"Train: {n_train} | Val: {n_val}")

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)
    val_dl   = DataLoader(val_ds  , batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)
    full_dl  = DataLoader(dataset , batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)

    # ============================================================
    # MODEL + OPTIMIZER + LOSS
    # ============================================================
    model = CNNGRU(image_size=IMAGE_SIZE, seq_len=SEQ_LEN, num_outputs=5).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler = GradScaler()

    criterion = MixedLoss(speed_weight=0.2)

    # ============================================================
    # METRICS
    # ============================================================
    def compute_metrics(logits, targets):
        # logits: raw outputs (B,5), targets: (B,5)
        probs = torch.sigmoid(logits[:, :4])  # only controls
        preds_bin = (probs > 0.5).float()
        targets_bin = (targets[:, :4] > 0.5).float()

        exact_match = (preds_bin == targets_bin).all(dim=1).float().mean().item()
        per_label_acc = (preds_bin == targets_bin).float().mean(dim=0).cpu().tolist()
        return exact_match, per_label_acc

    # ============================================================
    # VALIDATION LOOP
    # ============================================================
    def validation():
        model.eval()
        val_loss_sum = 0.0

        exact_sum = 0.0
        label_acc_sum = [0.0, 0.0, 0.0, 0.0]  # only controls

        with torch.no_grad():
            for seqs, labels in val_dl:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                logits = model(seqs)
                batch_size = seqs.size(0)

                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * batch_size

                ex, pla = compute_metrics(logits, labels)
                exact_sum += ex * batch_size
                label_acc_sum = [la + p * batch_size for la, p in zip(label_acc_sum, pla)]

        avg_val_loss = val_loss_sum / n_val if n_val > 0 else float("nan")
        exact = exact_sum / n_val if n_val > 0 else float("nan")
        per_label = [la / n_val for la in label_acc_sum] if n_val > 0 else [float("nan")] * 4
        return avg_val_loss, exact, per_label

    # ============================================================
    # TRAIN LOOP
    # ============================================================
    best_val = float("inf")
    no_improve = 0
    best_ckpt_path = None
    history = {"train": [], "val": [], "val_exact": [], "val_labels_acc": []}

    print("Starting training...")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for seqs, labels in train_dl:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            batch_size = seqs.size(0)

            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type):
                logits = model(seqs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            # unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        # scheduler step (after optimizer)
        scheduler.step()

        train_loss = train_loss_sum / train_samples if train_samples > 0 else float("nan")
        # Validation
        val_loss, val_exact, val_label = validation()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["val_exact"].append(val_exact)
        history["val_labels_acc"].append(val_label)

        # === Logging ===
        elapsed = time.time() - start
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{EPOCHS} - Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {elapsed:.1f}s"
                f"\texact={val_exact:.3f} | labels={[round(a,3) for a in val_label]}"
            )

        # === Save checkpoint on improvement ===
        if best_val - val_loss > MIN_DELTA:
            best_val = val_loss
            no_improve = 0
            ckpt_path = CHECKPOINT_DIR / f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_ckpt_path = ckpt_path
            print(f"  -> New best validation ({best_val:.6f}), saved: {ckpt_path}")
        else:
            no_improve += 1

        # === Early stopping ===
        if no_improve >= PATIENCE:
            print(f"Early stopping (no improvement for {PATIENCE} epochs).")
            break

    # ============================================================
    # Save final best model
    # ============================================================
    if best_ckpt_path is not None and best_ckpt_path.exists():
        # load best checkpoint to model (to ensure saved model is the best)
        state = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Best model saved to {MODEL_SAVE_PATH} (from {best_ckpt_path})")
    else:
        # no checkpoint saved (rare) -> save current model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Model saved to {MODEL_SAVE_PATH} (no checkpoint existed)")

    # ============================================================
    # Post-training: confusion matrices & plots
    # ============================================================
    print("Generating confusion matrices...")
    plot_confusion_matrices(model, full_dl, DEVICE)

    print("Generating train/val loss curve...")
    plot_train_curve(history)   # expects history dict like above

    print("Generating val metric curves...")
    plot_val_curve(history)

    print("Training finished.")


if __name__ == "__main__":
    train()
