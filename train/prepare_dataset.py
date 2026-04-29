"""
Dataset Preparation Script
---------------------------
Downloads or organizes the SDNET2018 / Concrete Crack Images dataset
into the required folder structure:

  dataset/
    train/
      cracked/        ← images of cracked concrete/walls/bridges
      non_cracked/    ← images of intact surfaces
    val/
      cracked/
      non_cracked/
    test/
      cracked/
      non_cracked/

USAGE:
  python train/prepare_dataset.py --source <path_to_raw_images> --split 0.7 0.15 0.15

OR: If you already placed images manually in dataset/, just run:
  python train/prepare_dataset.py --check
"""

import argparse
import os
import shutil
import random
from pathlib import Path

DATASET_ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]
CLASSES = ["cracked", "non_cracked"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def check_dataset():
    """Print a summary of how many images are in each split/class folder."""
    print("\n📊 Dataset Summary")
    print("=" * 40)
    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            folder = DATASET_ROOT / split / cls
            if folder.exists():
                count = len([
                    f for f in folder.iterdir()
                    if f.suffix.lower() in VALID_EXTENSIONS
                ])
                print(f"  {split:6} / {cls:12} → {count:5} images")
                total += count
            else:
                print(f"  {split:6} / {cls:12} → MISSING ⚠️")
    print("=" * 40)
    print(f"  TOTAL: {total} images\n")


def organize_from_source(source_dir: str, train_ratio=0.70, val_ratio=0.15):
    """
    Recursively scan source_dir for images.
    Expects subdirectories named 'cracked' and 'non_cracked' (or similar).
    Splits images into train/val/test automatically.
    """
    source = Path(source_dir)
    test_ratio = round(1.0 - train_ratio - val_ratio, 2)
    print(f"\n📁 Scanning: {source}")
    print(f"   Split → train:{train_ratio} / val:{val_ratio} / test:{test_ratio}\n")

    # Create output dirs
    for split in SPLITS:
        for cls in CLASSES:
            (DATASET_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        # Try to find source folder (flexible naming)
        candidate_names = [cls, cls.replace("_", ""), cls.replace("non_cracked", "intact")]
        src_folder = None
        for name in candidate_names:
            candidate = source / name
            if candidate.exists():
                src_folder = candidate
                break

        if src_folder is None:
            print(f"  ⚠️  Could not find source folder for class '{cls}' in {source}")
            print(f"      Expected one of: {candidate_names}")
            continue

        images = [
            f for f in src_folder.rglob("*")
            if f.suffix.lower() in VALID_EXTENSIONS
        ]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split_map = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, files in split_map.items():
            dest_dir = DATASET_ROOT / split / cls
            for i, img_path in enumerate(files):
                dest = dest_dir / f"{cls}_{i:05d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dest)
            print(f"  ✅ {split:6} / {cls:12} → {len(files)} images copied")

    print("\n✅ Dataset organized successfully!")
    check_dataset()


def main():
    parser = argparse.ArgumentParser(description="Prepare structural damage dataset")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to raw dataset folder containing cracked/ and non_cracked/")
    parser.add_argument("--split", type=float, nargs=2, default=[0.70, 0.15],
                        metavar=("TRAIN", "VAL"),
                        help="Train/Val split ratios (test gets remainder). Default: 0.70 0.15")
    parser.add_argument("--check", action="store_true",
                        help="Only check and print dataset summary")
    args = parser.parse_args()

    if args.check or args.source is None:
        check_dataset()
    else:
        organize_from_source(args.source, args.split[0], args.split[1])


if __name__ == "__main__":
    main()
