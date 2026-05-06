"""
Converts VisDrone DET annotations to YOLO format.
Filters to person classes only (pedestrian=1, people=2).

VisDrone format: x,y,w,h,score,category,truncation,occlusion
YOLO format:     class_id cx cy w h  (normalised 0-1)

Usage:
    python utils/convert_annotations.py \
        --src  data/visdrone_det/VisDrone2019-DET-train \
        --dst  data/yolo_dataset/train
"""

import argparse
import shutil
from pathlib import Path
import cv2


PERSON_IDS = {1, 2}

IGNORE_SCORE = 0


def convert_annotation(
    ann_path: Path,
    img_path: Path,
    dst_label_path: Path,
) -> int:
    """
    Convert one VisDrone annotation file to YOLO format.
    Returns number of person boxes written.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    H, W = img.shape[:2]

    lines_out = []

    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 8:
                continue

            x, y, w, h     = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score           = int(parts[4])
            category        = int(parts[5])

            if score == IGNORE_SCORE:
                continue
            if category not in PERSON_IDS:
                continue

            if w <= 0 or h <= 0:
                continue

            if w < 4 or h < 4:
                continue

            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines_out.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if lines_out:
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label_path, "w") as f:
            f.write("\n".join(lines_out))

    return len(lines_out)


def convert_split(src_dir: Path, dst_dir: Path):
    """Convert an entire VisDrone split to YOLO format."""
    img_dir = src_dir / "images"
    ann_dir = src_dir / "annotations"

    dst_img_dir   = dst_dir / "images"
    dst_label_dir = dst_dir / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    total_boxes  = 0
    total_images = 0
    skipped      = 0

    print(f"\n[Convert] {src_dir.name} → {dst_dir}")
    print(f"  Found {len(ann_files)} annotation files")

    for ann_path in ann_files:
        img_path = img_dir / ann_path.with_suffix(".jpg").name
        if not img_path.exists():
            skipped += 1
            continue

        dst_label_path = dst_label_dir / ann_path.name
        dst_img_path   = dst_img_dir   / img_path.name

        n_boxes = convert_annotation(ann_path, img_path, dst_label_path)

        if n_boxes > 0:
            shutil.copy2(img_path, dst_img_path)
            total_boxes  += n_boxes
            total_images += 1

    print(f"  Images with persons : {total_images}")
    print(f"  Total person boxes  : {total_boxes}")
    print(f"  Skipped (no image)  : {skipped}")
    print(f"  Avg boxes/image     : {total_boxes/max(1,total_images):.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="VisDrone DET split dir (contains images/ and annotations/)")
    p.add_argument("--dst", required=True,
                   help="Output YOLO split dir")
    args = p.parse_args()

    convert_split(Path(args.src), Path(args.dst))
    print("\n[Convert] Done.")


if __name__ == "__main__":
    main()