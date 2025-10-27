import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl

# ---------------------
# Dataset and DataLoader
# ---------------------
class CustomDataset(Dataset):
    def __init__(self, path):
        df = pl.scan_parquet(path)
        self.X = df.select(pl.exclude(['forward','backward','left','right','speed'])).collect().to_torch()
        self.y = df.select(pl.col(['forward','backward','left','right','speed'])).collect().to_torch()
        assert self.X.shape[1:] == [2,128,128] # X.shape = [n,2,128,128]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)