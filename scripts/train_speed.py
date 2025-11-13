# train_with_speed.py
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

from datasets import CustomDataset   # assumes returns (images, labels) where labels = [f,b,l,r,speed]
from model import Autopilot         # expected to return (B,5) -> 4 logits + 1 speed (raw)

# ------------------------
# CONFIG
# ------------------------
PARQUET_PATH = "processed/dataset.parquet"
IMAGE_SIZE = (256, 256)
SEQ_LEN = 5

BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-6
NUM_WORKERS = 0

PATIENCE = 10
MIN_DELTA = 1e-4

SPEED_WEIGHT = 0.1

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# speed normalization constants
SPEED_MAX = 50.0   # user confirmed ±50
def norm_speed(s): return s / SPEED_MAX
def denorm_speed(s): return s * SPEED_MAX

# ------------------------
# OPTIONAL: Focal loss for arrows (recommended for imbalance)
# ------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = None if alpha is None else torch.tensor(alpha, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)  # pt = sigmoid for correct predictions
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ------------------------
# DATA
# ------------------------
print("Loading dataset...")
full_ds = CustomDataset(PARQUET_PATH, seq_len=SEQ_LEN)
n = len(full_ds)
train_len = int(0.8 * n)
train_ds = Subset(full_ds, list(range(train_len)))
val_ds   = Subset(full_ds, list(range(train_len, n)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"Total sequences: {n} | Train: {train_len} | Val: {n - train_len}")

# ------------------------
# Compute class frequencies from UNSMOOTHED labels (for monitoring / alpha)
# ------------------------
counts = torch.zeros(4)
total = 0
for _, y in full_ds:
    lbls = y[:4]  # first 4
    lbls_bin = (lbls > 0.5).float()
    counts += lbls_bin
    total += 1
freqs = counts / total
print("Class frequencies (unsmoothed):", freqs.tolist())

# Build alpha for focal: higher weight for rare classes
alpha = (1.0 - freqs).cpu().numpy()
alpha = alpha / (alpha.sum() + 1e-12) * len(alpha)  # normalize to roughly sum=4 so scale sensible
alpha = alpha.tolist()
print("Focal alpha (per-label):", alpha)

# ------------------------
# MODEL, LOSS, OPTIM
# ------------------------
base_model = Autopilot(image_size=IMAGE_SIZE, seq_len=SEQ_LEN).to(DEVICE)

# If model does not produce 5 outputs, wrap it with a small head
# We detect output size from a dummy forward pass.
with torch.no_grad():
    model_ok = True
    try:
        xd, yd = full_ds[0]
        x_dummy = torch.tensor(np.expand_dims(xd, 0)).to(DEVICE, dtype=torch.float32)
        out = base_model(x_dummy)
        out_shape = tuple(out.shape)
        if out_shape[-1] == 5:
            model = base_model
            print("Autopilot outputs 5 dims (4 logits + 1 speed). Using it directly.")
        elif out_shape[-1] == 4:
            # attach linear head to predict speed
            hidden_dim = out_shape[-1]
            class WrapperModel(nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base
                    self.speed_head = nn.Linear(hidden_dim, 1)
                    # initialize small weights
                    nn.init.normal_(self.speed_head.weight, std=1e-3)
                    nn.init.constant_(self.speed_head.bias, 0.0)
                def forward(self, x):
                    logits = self.base(x)          # (B,4)
                    speed = self.speed_head(logits)  # (B,1)
                    return torch.cat([logits, speed], dim=1)
            model = WrapperModel(base_model).to(DEVICE)
            print("Autopilot returned 4 dims — wrapped with a speed head.")
        else:
            model_ok = False
            print("Autopilot returned unexpected output shape:", out_shape)
    except Exception as e:
        print("Warning during model shape probe:", e)
        # Fallback: assume model gives 5-dim output at runtime
        model = base_model

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
scaler = GradScaler()

# losses
#arrow_loss_fn = FocalLoss(gamma=2.0, alpha=alpha, reduction="mean")   # multi-label
weight = [
    0.36, #forward
    34, #back
    2, #left
    1.4 #right
]
pos_weight = torch.log1p(torch.tensor(weight).to(DEVICE)).to(DEVICE, dtype=torch.float32)
arrow_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
speed_loss_fn = nn.SmoothL1Loss(reduction="mean")                     # robust regression

# ------------------------
# Utilities: metrics, validation
# ------------------------
def compute_losses_and_metrics(logits5, targets5):
    """
    logits5: (B,5) -> first 4 are logits for arrows, last is speed_pred (unbounded)
    targets5: (B,5) -> first 4 are binary (0/1), last is speed in original scale [-50,50]
    """
    arrows_logits = logits5[:, :4]
    speed_pred_raw = logits5[:, 4]
    arrows_targets = targets5[:, :4]
    speed_targets = targets5[:, 4]

    # normalize speed to [-1, 1] range for loss stability
    speed_targets_norm = speed_targets / SPEED_MAX
    speed_pred_norm = speed_pred_raw

    loss_arrows = arrow_loss_fn(arrows_logits, arrows_targets)
    loss_speed = speed_loss_fn(speed_pred_norm, speed_targets_norm)

    loss = loss_arrows + SPEED_WEIGHT * loss_speed

    # metrics
    with torch.no_grad():
        probs = torch.sigmoid(arrows_logits)
        preds_bin = (probs > 0.5).float()
        targets_bin = (arrows_targets > 0.5).float()

        exact_match = (preds_bin == targets_bin).all(dim=1).float().mean().item()
        per_label_acc = (preds_bin == targets_bin).float().mean(dim=0).cpu().numpy().tolist()

        # speed metrics (denormalize) — keep computations in torch
        # ensure denorm_speed returns a tensor; if it returns numpy, convert back with torch.from_numpy(...)
        speed_pred_den = denorm_speed(speed_pred_norm.detach().cpu())   # should be tensor
        speed_t_den = denorm_speed(speed_targets_norm.detach().cpu())

        if not torch.is_tensor(speed_pred_den):
            speed_pred_den = torch.as_tensor(speed_pred_den)
        if not torch.is_tensor(speed_t_den):
            speed_t_den = torch.as_tensor(speed_t_den)

        diff = speed_pred_den - speed_t_den
        speed_mse = (diff ** 2).mean().item()
        speed_rmse = np.sqrt(speed_mse)
        speed_mae = diff.abs().mean().item()

    return loss, loss_arrows.item(), loss_speed.item(), exact_match, per_label_acc, speed_rmse, speed_mae

def validate():
    model.eval()
    running_loss = 0.0
    running_loss_ar = 0.0
    running_loss_sp = 0.0
    n = 0
    exact_sum = 0.0
    per_label_sum = np.zeros(4)
    speed_rmse_sum = 0.0
    speed_mae_sum = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)

            logits5 = model(xb)
            loss, lar, lsp, exact, pla, sprmse, spmae = compute_losses_and_metrics(logits5, yb)
            b = xb.size(0)
            running_loss += loss.item() * b
            running_loss_ar += lar * b
            running_loss_sp += lsp * b
            n += b
            exact_sum += exact * b
            per_label_sum += np.array(pla) * b
            speed_rmse_sum += sprmse * b
            speed_mae_sum += spmae * b

    avg_loss = running_loss / n
    avg_ar_loss = running_loss_ar / n
    avg_sp_loss = running_loss_sp / n
    avg_exact = exact_sum / n
    avg_label_acc = (per_label_sum / n).tolist()
    avg_speed_rmse = speed_rmse_sum / n
    avg_speed_mae = speed_mae_sum / n

    return avg_loss, avg_ar_loss, avg_sp_loss, avg_exact, avg_label_acc, avg_speed_rmse, avg_speed_mae

# ------------------------
# Training loop
# ------------------------
best_val = float("inf")
no_improve = 0
start_time = time.time()

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_ar_loss = 0.0
    epoch_sp_loss = 0.0
    batches = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE, dtype=torch.float32)
        yb = yb.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()
        with autocast(device_type=str(DEVICE)):
            logits5 = model(xb)   # (B,5)
            loss, lar, lsp, _, _, _, _ = compute_losses_and_metrics(logits5, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_ar_loss += lar
        epoch_sp_loss += lsp
        batches += 1

    train_loss = epoch_loss / max(1, batches)
    train_ar_loss = epoch_ar_loss / max(1, batches)
    train_sp_loss = epoch_sp_loss / max(1, batches)

    val_loss, val_ar_loss, val_sp_loss, val_exact, val_label_acc, val_sp_rmse, val_sp_mae = validate()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} (ar={train_ar_loss:.4f},sp={train_sp_loss:.4f})"
              f" | val_loss={val_loss:.4f} (ar={val_ar_loss:.4f},sp={val_sp_loss:.4f})"
              f" | exact={val_exact:.3f} | labels={[round(a,3) for a in val_label_acc]} "
              f"| speed_rmse={val_sp_rmse:.3f} speed_mae={val_sp_mae:.3f}")

    # checkpointing and early stopping
    improved = (best_val - val_loss) > MIN_DELTA
    if improved:
        best_val = val_loss
        no_improve = 0
        ckpt = CHECKPOINT_DIR / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val": best_val
        }, ckpt)
        print(f"  Saved best checkpoint: {ckpt}")
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print("Early stopping triggered.")
        break

    scheduler.step()

print("Training finished in", round(time.time() - start_time, 1), "seconds.")
print("Best val:", best_val)

# ------------------------
# Utilities: threshold search and confusion matrix
# ------------------------
def find_best_thresholds(loader, device=DEVICE, label_names=None):
    model.eval()
    probs_list = []
    targets_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)
            out = model(xb)
            probs_list.append(torch.sigmoid(out[:, :4]).cpu().numpy())
            targets_list.append((yb[:, :4] > 0.5).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    n_labels = probs.shape[1]
    best_thresh = np.zeros(n_labels, dtype=float)
    for i in range(n_labels):
        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.01, 0.99, 99):
            p = (probs[:, i] >= t).astype(int)
            tp = int(((p == 1) & (targets[:, i] == 1)).sum())
            fp = int(((p == 1) & (targets[:, i] == 0)).sum())
            fn = int(((p == 0) & (targets[:, i] == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresh[i] = best_t
        name = label_names[i] if label_names else f"label_{i}"
        print(f"{name}: best_thresh={best_t:.2f} best_f1={best_f1:.4f}")
    return best_thresh

def compute_confusion_matrices(loader, thresholds=0.5, label_names=None):
    model.eval()
    if label_names is None:
        label_names = [f"label_{i}" for i in range(4)]
    num_labels = len(label_names)
    conf = np.zeros((num_labels, 2, 2), dtype=np.int64)  # [[TP,FP],[FN,TN]]
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)
            out = model(xb)
            probs = torch.sigmoid(out[:, :4]).cpu().numpy()
            targets = (yb[:, :4].cpu().numpy() > 0.5).astype(int)

            pred_bin = (probs >= thresholds.reshape(1, -1)).astype(int) if hasattr(thresholds, "reshape") else (probs >= thresholds).astype(int)
            B = pred_bin.shape[0]
            total += B
            for i in range(num_labels):
                p = pred_bin[:, i]
                t = targets[:, i]
                TP = int(((p == 1) & (t == 1)).sum())
                FP = int(((p == 1) & (t == 0)).sum())
                FN = int(((p == 0) & (t == 1)).sum())
                TN = int(((p == 0) & (t == 0)).sum())
                conf[i, 0, 0] += TP
                conf[i, 0, 1] += FP
                conf[i, 1, 0] += FN
                conf[i, 1, 1] += TN

    print(f"\nComputed confusion matrices on {total} samples")
    for i, name in enumerate(label_names):
        TP, FP = conf[i, 0]
        FN, TN = conf[i, 1]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"\nLabel: {name}")
        print(f"  [[TP {TP:6d}  FP {FP:6d}]")
        print(f"   [FN {FN:6d}  TN {TN:6d}]]")
        print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

# ------------------------
# Final evaluation: thresholds + confusion + speed errors
# ------------------------
label_names = ["forward", "back", "left", "right"]
print("\nFinding best thresholds per label on validation set (F1 maximization)...")
best_thresh = find_best_thresholds(val_loader, device=DEVICE, label_names=label_names)
best_thresh = 0.5
print("\nUsing thresholds:", best_thresh)

compute_confusion_matrices(full_loader, thresholds=best_thresh, label_names=label_names)

# compute final speed metrics on val set
def speed_eval(loader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)
            out = model(xb)
            sp_pred = out[:, 4].cpu().numpy()
            sp_t = yb[:, 4].cpu().numpy()
            preds.append(sp_pred)
            targets.append(sp_t)
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    # preds assumed normalized? if model learned normalized speed, denormalize:
    preds_den = denorm_speed(preds)
    targets_den = targets  # original scale already ±50
    rmse = float(np.sqrt(np.mean((preds_den - targets_den) ** 2)))
    mae = float(np.mean(np.abs(preds_den - targets_den)))
    return rmse, mae

rmse, mae = speed_eval(val_loader)
print(f"\nSpeed eval on val set: RMSE={rmse:.3f}  MAE={mae:.3f} (units: same as dataset, ±{SPEED_MAX})")
print(pos_weight)

# Save final model
last_ckpt = CHECKPOINT_DIR / "last_model.pt"
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "best_val": best_val
}, last_ckpt)
print("Saved final checkpoint:", last_ckpt)
