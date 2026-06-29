from __future__ import annotations

"""
Split Potato dataset with YOLOv8 auto-crop and stratified train/valid/test CSV.

Flow:
- Scan POTATO_RAW_DIR/<disease> for images.
- For each image, run yolov8n.pt; crop the highest-confidence box and save to POTATO_CLEAN_DIR/<disease>/<filename>.
- If YOLO finds nothing or the crop is invalid, copy the original image instead.
- Skip processing if the cropped file already exists (resumable).
- Build a CSV with image paths relative to POTATO_CLEAN_DIR (columns: image, disease, plant_type, dataset_type).
"""

import shutil
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


# =============================
# Path configuration (Windows)
# =============================
# raw images organized as: POTATO_RAW_DIR/<disease>/*.jpg
POTATO_RAW_DIR = Path(
    r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\dataset\Potato Leaf Disease Dataset in Uncontrolled Environment"
)

# Model + versioning config
# Use custom-trained expert model to detect leaves (ignores fingers/noise)
YOLO_MODEL_PATH = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\models\detection\yolo11m_v2.pt"
VERSION_TAG = "v2_expert"
CONF_THRESHOLD = 0.6  # high confidence since model is specialized

# Cropped YOLO images destination (versioned)
POTATO_CLEAN_DIR = Path(
    fr"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\dataset\potato_dataset_clean"
)

# Log file for skipped images (no detection)
SKIPPED_LOG = POTATO_CLEAN_DIR / "skipped_images.txt"

# Where to write the CSV output
CSV_OUTPUT_DIR = Path(
    r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\plant_health_app\data"
)

# Output file name (versioned)
OUTPUT_CSV_NAME = f"potato_data_{VERSION_TAG}.csv"

# Split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) < 1e-8, "Split ratios must sum to 1.0"

RANDOM_STATE = 42
PLANT_TYPE = "Potato"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def ensure_cropped_image(raw_path: Path, disease: str, model: YOLO) -> Path:
    """Crop a single image with YOLO; fallback to copying original."""

    out_dir = POTATO_CLEAN_DIR / disease
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / raw_path.name

    # Skip if already processed
    if out_path.exists():
        return out_path

    img = cv2.imread(str(raw_path))
    if img is None:
        print(f"[WARN] Unable to read image, copying as-is: {raw_path}")
        shutil.copyfile(raw_path, out_path)
        return out_path

    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
    best_crop = None

    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = int(confidences.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()

        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))

        if x2 > x1 and y2 > y1:
            best_crop = img[y1:y2, x1:x2]

    if best_crop is not None and best_crop.size > 0:
        cv2.imwrite(str(out_path), best_crop)
        return out_path

    if best_crop is not None and best_crop.size > 0:
        cv2.imwrite(str(out_path), best_crop)
        return out_path

    # No detection: log and copy original to keep dataset usable
    SKIPPED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SKIPPED_LOG.open("a", encoding="utf-8") as f:
        f.write(str(raw_path) + "\n")

    shutil.copyfile(raw_path, out_path)
    return out_path


def collect_images(raw_dir: Path, model: YOLO) -> List[dict]:
    """Scan raw_dir/<disease>, crop each image, and build rows with cleaned paths."""
    rows: List[dict] = []
    if not raw_dir.exists():
        raise FileNotFoundError(f"POTATO_RAW_DIR not found: {raw_dir}")

    for disease_dir in sorted(raw_dir.iterdir()):
        if not disease_dir.is_dir():
            continue
        disease = disease_dir.name
        for file in disease_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTS:
                cleaned_path = ensure_cropped_image(file, disease, model)
                rel_path = cleaned_path.relative_to(POTATO_CLEAN_DIR)
                rows.append(
                    {
                        "image": rel_path.as_posix(),
                        "disease": disease,
                        "plant_type": PLANT_TYPE,
                    }
                )
    if not rows:
        raise ValueError(f"No images found under: {raw_dir}")
    return rows


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split DataFrame into train/valid/test with stratification on disease."""
    train_df, temp_df = train_test_split(
        df,
        test_size=VALID_RATIO + TEST_RATIO,
        stratify=df["disease"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
        stratify=temp_df["disease"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    train_df["dataset_type"] = "train"
    valid_df["dataset_type"] = "valid"
    test_df["dataset_type"] = "test"

    return pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)


def main() -> None:
    model = YOLO(YOLO_MODEL_PATH)

    rows = collect_images(POTATO_RAW_DIR, model)
    df = pd.DataFrame(rows, columns=["image", "disease", "plant_type"])

    split_df = stratified_split(df)

    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CSV_OUTPUT_DIR / OUTPUT_CSV_NAME
    split_df.to_csv(output_path, index=False)

    print(f"Saved {len(split_df)} rows to {output_path}")
    print(split_df["dataset_type"].value_counts())


if __name__ == "__main__":
    main()
