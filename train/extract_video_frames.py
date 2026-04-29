"""
Video Frame Extractor for Training
────────────────────────────────────────────────────────
Extracts frames from crack/non-crack videos into the dataset folder.

USAGE:
  # Extract from a folder of cracked videos:
  python train/extract_video_frames.py --source C:\path\to\crack_videos --label cracked

  # Extract from non-cracked videos:
  python train/extract_video_frames.py --source C:\path\to\normal_videos --label non_cracked

  # Extract every Nth frame (default: every 15 frames = 2fps from 30fps video):
  python train/extract_video_frames.py --source C:\videos --label cracked --every 10

STRUCTURE expected in --source folder:
  source/
    video1.mp4
    video2.avi
    ...

OUTPUT goes into:
  dataset/train/cracked/     or    dataset/train/non_cracked/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

SUPPORTED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def extract(source_dir: str, label: str, every: int, out_dir: Path):
    source = Path(source_dir)
    if not source.exists():
        print(f"❌ Source not found: {source}")
        sys.exit(1)

    videos = [f for f in source.iterdir() if f.suffix.lower() in SUPPORTED]
    if not videos:
        print(f"❌ No video files found in {source}")
        print(f"   Supported: {SUPPORTED}")
        sys.exit(1)

    dest = out_dir / "dataset" / "train" / label
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n📹 Extracting frames")
    print(f"   Source    : {source}")
    print(f"   Label     : {label}")
    print(f"   Every     : {every} frames")
    print(f"   Output    : {dest}")
    print(f"   Videos    : {len(videos)}\n")

    total_saved = 0
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ⚠️  Cannot open: {video_path.name} — skipping")
            continue

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30
        n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = round(n_frames / fps, 1)
        saved      = 0
        frame_idx  = 0

        print(f"  📄 {video_path.name} ({duration}s | {n_frames} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every == 0:
                # Resize to 256x256 (model uses 224x224 — slight padding is fine)
                frame_resized = cv2.resize(frame, (256, 256))
                fname = dest / f"{video_path.stem}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(fname), frame_resized,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
            frame_idx += 1

        cap.release()
        total_saved += saved
        print(f"     → {saved} frames saved")

    print(f"\n✅ Done — {total_saved} total frames saved to:")
    print(f"   {dest}")
    print(f"\nNext step: python train/train_model.py --epochs 15 --batch 8")


def main():
    parser = argparse.ArgumentParser(description="Extract training frames from videos")
    parser.add_argument("--source",  required=True, help="Folder containing video files")
    parser.add_argument("--label",   required=True, choices=["cracked", "non_cracked"],
                        help="Class label for these videos")
    parser.add_argument("--every",   type=int, default=15,
                        help="Extract every Nth frame (default: 15 = ~2fps from 30fps)")
    parser.add_argument("--outdir",  default=".",
                        help="Project root folder (default: current directory)")
    args = parser.parse_args()

    extract(
        source_dir = args.source,
        label      = args.label,
        every      = args.every,
        out_dir    = Path(args.outdir),
    )


if __name__ == "__main__":
    main()
