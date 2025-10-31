import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl

# ---------------------
# Dataset and DataLoader
# ---------------------
class CustomDataset(Dataset):
    def __init__(self, path, sequence_length: int):
        df = pl.scan_parquet(path)  # use read_parquet to load eagerly
        self.X = df.select(pl.exclude(['forward','backward','left','right','speed'])).collect().to_torch()
        self.y = df.select(['forward','backward','left','right','speed']).collect().to_torch()
        self.sequence_length = sequence_length

        # Reshape flat images to [128, 128]
        self.X = self.X.reshape(-1, 128, 128)

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of images and add channel dim: [seq_len, 1, 128, 128]
        img_seq = self.X[idx : idx + self.sequence_length]
        img_seq = torch.tensor(img_seq, dtype=torch.float32).unsqueeze(1)

        # Get target control + speed
        target = torch.tensor(self.y[idx + self.sequence_length], dtype=torch.float32)

        return img_seq, target
