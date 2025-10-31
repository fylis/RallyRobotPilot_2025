import itertools
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, SequentialSampler

from model import RallyAutopilot
from datasets import CustomDataset

BATCH_SIZE = 32
NUM_EPOCHS = 100
MIN_DELTA = 1e-4
PATIENCE = 10
SAVE_CHECKPOINT_DIR = "checkpoints"
DATA_FILE = "data/data.parquet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RallyAutopilot().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

scaler = GradScaler()  # for mixed precision

full_ds = CustomDataset(DATA_FILE, 5)

train_len = int(0.8 * len(full_ds))
train_ds = Subset(full_ds, range(0, train_len))
val_ds   = Subset(full_ds, range(train_len, len(full_ds)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(train_ds))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(val_ds))

def loss(true_outputs, pred_outputs):
    # layout: [forward, break, left, right, speed]
    output_cmd = pred_outputs[:, :4]   # binary
    y_cmd = true_outputs[:,:4]
    output_speed = pred_outputs[:, 4:5]       # regression target
    y_speed = true_outputs[:,4:5]

    # Compute loss
    loss_cmd = nn.BCEWithLogitsLoss()(output_cmd, y_cmd)# Binary commands

    loss_speed = nn.MSELoss()(output_speed, y_speed)

    # optional auxiliary raycast loss
    # pred_raycast = pred_outputs[:, 5:5+R]
    # loss_raycast = nn.MSELoss()(pred_raycast, target_raycast.to(device))

    total_loss = loss_cmd + loss_speed   # weight losses as needed
    return total_loss

def save_checkpoint(state, fname):
    torch.save(state, fname)

def load_checkpoint(fname, model_load, optimizer=optimizer, scheduler=scheduler):
    ckpt = torch.load(fname, map_location=device)
    model_load.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt.get("epoch", -1), ckpt.get("best_val", None)

def validate(model_val, loader, device = device):
    model_val.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model_val(xb)
            preds.append(out)
            ys.append(yb)
    y_true = torch.cat(ys, axis=0)
    y_pred = torch.cat(preds, axis=0)
    return loss(y_true, y_pred)

best_val = float("inf")
best_epoch = -1
no_improve = 0

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast():
            B, S, C, H, W = xb.shape
            outputs = model(xb)
            epoch_loss = loss(yb, outputs)
        
        epoch_losses.append(epoch_loss.detach())

        scaler.scale(epoch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

    train_loss = torch.mean(torch.Tensor(epoch_losses))
    val_loss = validate(model,val_loader)
    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.5f} | val_rmse {val_loss:.5f} | time {elapsed:.1f}s")

    # early stopping & checkpoint best
    improved = (best_val - val_loss) > MIN_DELTA
    if improved:
        best_val = val_loss
        best_epoch = epoch
        no_improve = 0
        ckpt_path = Path(SAVE_CHECKPOINT_DIR) / f"best_epoch_{epoch:03d}.pt"
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val": best_val
        }, str(ckpt_path))
        print(f"  Saved best checkpoint to {ckpt_path} {best_val}")
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print(f"Early stopping triggered. No improvement for {no_improve} epochs. Best val {best_val:.5f} at epoch {best_epoch}.")
        break

    scheduler.step()
