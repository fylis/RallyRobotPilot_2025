import numpy as np
import cv2
import polars as pl
import os

DATA_DIR = 'records/'
DATA_SAVE = "data/data.parquet"

def preprocess_image(image, crop_top=0, crop_bottom=0, target_size=(128, 128)):
    # Optional cropping (e.g., remove top 50px and bottom 20px)
    if crop_top > 0 or crop_bottom > 0:
        image = image[crop_top:image.shape[0]-crop_bottom, :, :]
    

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # shape: (H, W)

    # Resize to target size
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized  # shape: (128, 128)

def load_and_preprocess_image(path):
    # Load RGB image from .npy
    img = np.load(path)  # shape: (H, W, 3)
    return preprocess_image(img)

def encode_controls(record):
    return np.array([
        record["forward"],
        record["back"],
        record["left"],
        record["right"]
    ], dtype=np.float32) #binary

def build_dataset(record_file):
    records = np.load(record_file, allow_pickle=True)
    images = []
    controls = []
    speeds = []

    for record in records:
        img = load_and_preprocess_image(record["image"])
        ctrl = encode_controls(record)
        spd = np.array([record["car_speed"]], dtype=np.float32)

        images.append(img)
        controls.append(ctrl)
        speeds.append(spd)

    return np.array(images), np.array(controls), np.array(speeds)

def save_to_parquet(X, y, path):
    flat_X = X.reshape(X.shape[0], -1)
    df = pl.DataFrame(flat_X.tolist(), schema=[f"px_{i}" for i in range(flat_X.shape[1])])
    df = df.with_columns([
        pl.Series("forward", y[:, 0]),
        pl.Series("backward", y[:, 1]),
        pl.Series("left", y[:, 2]),
        pl.Series("right", y[:, 3]),
        pl.Series("speed", y[:, 4]),
    ])
    df.write_parquet(path)

def main():
    ld_save = pl.LazyFrame()
    files = os.listdir(DATA_DIR)
    for filename in files:
        if filename == 'images':
            continue
        input_path = os.path.join(DATA_DIR, filename)
        if not filename.endswith(".npy"):
            if os.path.isdir(input_path):
                print(filename)
                fs = os.listdir(input_path)
                for f in fs:
                    files.append(os.path.join(filename,f))
            continue
        print(filename)
        input_path = os.path.join(DATA_DIR, filename)

        # Build list of records
        imgs, ctrs, speeds = build_dataset(input_path)

        flat_X = imgs.reshape(imgs.shape[0], -1)
        df = pl.DataFrame(flat_X.tolist(), schema=[f"px_{i}" for i in range(flat_X.shape[1])])

        # Combine ctrs and speeds into one target array
        y = np.concatenate([ctrs, speeds], axis=1)

        df = df.with_columns([
            pl.Series("forward", y[:, 0]),
            pl.Series("backward", y[:, 1]),
            pl.Series("left", y[:, 2]),
            pl.Series("right", y[:, 3]),
            pl.Series("speed", y[:, 4]),
        ])

        ld_save = pl.concat([ld_save, df.lazy()], how='diagonal')

    ld_save.sink_parquet(DATA_SAVE)

if  __name__ == "__main__":
    main()