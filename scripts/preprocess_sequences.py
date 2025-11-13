# preprocess_sequences_mp.py
import os
import numpy as np
import polars as pl
import cv2
import lzma
import pickle
from multiprocessing import Pool, cpu_count

# === CONFIG ===
IMAGE_SIZE = (256, 256)
RAW_DATA_DIR = "data"
OUT_IMAGE_DIR = "processed/images"
OUT_PARQUET = "processed/dataset_sequences.parquet"
SEQ_LEN = 5   # Number of frames per sequence
NUM_WORKERS = max(1, cpu_count() - 1)  # Use all but one core

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)


# ---------- IMAGE PREPROCESS ----------
def preprocess_image(img, image_size=IMAGE_SIZE):
    """Convert RGB to grayscale, resize, normalize to [0,1]."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ---------- LOAD ONE NPZ FILE ----------
def process_npz_file(npz_path):
    """Load compressed pickle file, process each frame, and return frame dicts."""
    try:
        with lzma.open(npz_path, "rb") as f:
            snapshots = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load {npz_path}: {e}")
        return []

    if not isinstance(snapshots, list):
        print(f"⚠️ {npz_path} does not contain a list of snapshots.")
        return []

    base = os.path.splitext(os.path.basename(npz_path))[0]
    rows = []

    for i, s in enumerate(snapshots):
        try:
            forward = float(s.current_controls[0])
            back    = float(s.current_controls[1])
            left    = float(s.current_controls[2])
            right   = float(s.current_controls[3])
            speed   = float(s.car_speed)
            img_raw = s.image
        except Exception as e:
            print(f"⚠️ Skipped corrupted entry in {npz_path}: {e}")
            continue

        rel_path = os.path.relpath(npz_path, RAW_DATA_DIR).replace(os.sep, "_")
        base = os.path.splitext(rel_path)[0]

        img = preprocess_image(img_raw)
        out_img_path = os.path.join(OUT_IMAGE_DIR, f"{base}_{i}.npy")
        np.save(out_img_path, img)

        rows.append({
            "image_path": out_img_path,
            "forward": forward,
            "back": back,
            "left": left,
            "right": right,
            "speed": speed
        })

    return rows


# ---------- MAKE SEQUENCES ----------
def build_sequences(all_rows, seq_len=SEQ_LEN):
    """Convert flat list of frames into overlapping fixed-length sequences."""
    sequences = []
    for i in range(len(all_rows) - seq_len + 1):
        seq = all_rows[i:i + seq_len]
        entry = {f"image_path_{j}": seq[j]["image_path"] for j in range(seq_len)}
        # use label of last frame
        entry.update({
            "forward": seq[-1]["forward"],
            "back": seq[-1]["back"],
            "left": seq[-1]["left"],
            "right": seq[-1]["right"],
            "speed": seq[-1]["speed"],
        })
        sequences.append(entry)
    return sequences


# ---------- MAIN ----------
def main():
    # Collect all .npz files
    npz_files = []
    for root, _, files in os.walk(RAW_DATA_DIR):
        for f in files:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(root, f))

    if not npz_files:
        print("⚠️ No .npz files found in data directory.")
        return

    print(f"🔍 Found {len(npz_files)} files. Using {NUM_WORKERS} workers...")

    # Parallel file processing
    with Pool(processes=NUM_WORKERS) as pool:
        all_results = pool.map(process_npz_file, npz_files)

    # Build sequences per file
    all_sequences = []
    all_rows = []
    total_rows = 0

    for rows in all_results:
        if rows and len(rows) >= SEQ_LEN:
            seqs = build_sequences(rows, SEQ_LEN)
            all_sequences.extend(seqs)
            total_rows += len(rows)
        for r in rows:
            all_rows.append(r)

    if not all_sequences:
        print("⚠️ Not enough valid sequences to save.")
        return

    # Add index
    for i, seq in enumerate(all_sequences):
        seq["index"] = i

    # Save dataset
    df = pl.DataFrame(all_sequences)
    df.write_parquet(OUT_PARQUET)

    print(f"✅ Saved sequence dataset: {OUT_PARQUET}")
    print(f"✅ Total sequences: {len(all_sequences)}")
    print(f"✅ Total frames: {len(all_rows)}")


if __name__ == "__main__":
    main()
