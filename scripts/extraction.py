import numpy as np
import lzma
import pickle
import os

RECORD_DIR = 'data/'
SAVE_DIR = 'records/'
IMAGE_DIR = 'records/images'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

files = os.listdir(RECORD_DIR)
for filename in files:
    input_path = os.path.join(RECORD_DIR, filename)
    if not filename.endswith(".npz"):
        if os.path.isdir(input_path):
            fs = os.listdir(input_path)
            for f in fs:
                files.append(os.path.join(filename,f))
        continue

    try:
        with lzma.open(input_path, "rb") as f:
            snapshots = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        continue

    base, ext = os.path.splitext(filename)
    # Build list of records
    records = []
    for idx, s in enumerate(snapshots):
        image_filename = f"{base}_{idx}.npy"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
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
        record_path = os.path.join(SAVE_DIR, base + ".npy")
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        np.save(record_path, records)
        print(f"✅ Converted {filename} → {record_path}")
    except Exception as e:
        print(f"❌ Error converting {filename}: {e}")