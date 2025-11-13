import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl

# === GLOBAL CONFIG ===
IMAGE_SIZE = (256, 256)   # must match preprocess_sequences.py


class CustomDataset(Dataset):
    """
    Loads labels from a Parquet file and image data from .npy files.
    Each row in the parquet represents one sequence of SEQ_LEN images.
    """

    def __init__(self, parquet_path: str):
        # Load Parquet into memory once (Polars -> fast)
        df = pl.read_parquet(parquet_path)
        self.df = df

        # Extract label tensors
        label_names = ["forward", "back", "left", "right", "speed"]
        self.labels = torch.tensor(
            df.select(label_names).to_numpy(),
            dtype=torch.float32
        )  # shape (N, 5)

        # Get all image path columns (sorted)
        image_columns = sorted([c for c in df.columns if c.startswith("image_path_")])

        # Convert to numpy for fast indexing
        self.image_paths = df.select(image_columns).to_numpy()  # shape (N, seq_len)
        self.seq_len = self.image_paths.shape[1]
        self.n = self.image_paths.shape[0]

        if self.n == 0:
            raise ValueError("Empty dataset — check your parquet file.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Load sequence of npy grayscale images
        frames = []
        for t in range(self.seq_len):
            img_path = self.image_paths[idx, t]
            img = np.load(img_path, mmap_mode="r")  # shape (H, W)
            img = np.expand_dims(img, axis=0)       # -> (1, H, W)
            frames.append(img)

        # Stack into (seq_len, 1, H, W)
        seq = np.stack(frames, axis=0)
        seq_tensor = torch.from_numpy(seq).float()  # (seq_len, 1, H, W)

        # Label for this sequence
        label_tensor = self.labels[idx]             # (5,)

        return seq_tensor, label_tensor
