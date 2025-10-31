import numpy as np
import polars as pl
import lzma
import pickle
import os

RECORD_DIR = 'records/'
SAVE_DIR = 'data/'
IMAGE_DIR = 'data/images'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

for filename in os.listdir(RECORD_DIR):
    if not filename.endswith(".npz"):
        continue

    input_path = os.path.join(RECORD_DIR, filename)
    try:
        with lzma.open(input_path, "rb") as f:
            snapshots = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        continue

    # Build list of records
    records = []
    for idx, s in enumerate(snapshots):
        image_filename = f"{filename}_{idx}.npy"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        np.save(image_path, s.image)
        record = {
            "forward": s.current_controls[0],
            "back": s.current_controls[1],
            "left": s.current_controls[2],
            "right": s.current_controls[3],
            "car_speed": s.car_speed,
            "image": image_path
        }
        records.append(record)
    try:
        record_path = os.path.join(SAVE_DIR, filename)
        np.save(record_path, records)
        print(f"✅ Converted {filename} → {record_path}")
    except Exception as e:
        print(f"❌ Error converting {filename}: {e}")