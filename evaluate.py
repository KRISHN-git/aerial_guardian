"""
Evaluates a trained model on the VisDrone val set.
Compares fine-tuned vs baseline on key metrics.

Usage:
    python evaluate.py --weights weights/finetune/drone_person_v1/weights/best.pt
    python evaluate.py --weights yolov8s.pt  (baseline comparison)
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from models.detector import DroneDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    required=True)
    p.add_argument("--data",       default="data/drone_person.yaml")
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--conf",       type=float, default=0.15)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--max-frames", type=int, default=200,
                   help="Frames for FPS benchmark")
    p.add_argument("--input",      default="data/visdrone/seq_01.mp4",
                   help="Video for FPS benchmark")
    return p.parse_args()


def run_official_val(weights: str, data: str, imgsz: int) -> dict:
    """Run Ultralytics official validation — gives mAP50, mAP50-95, precision, recall."""
    print(f"\n[Eval] Running official validation...")
    model   = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        conf=0.001,
        iou=0.6,
        verbose=False,
    )
    return {
        "mAP50":    float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }


def run_fps_benchmark(
    weights: str,
    input_path: str,
    imgsz: int,
    conf: float,
    max_frames: int,
) -> dict:
    """Measure real-world FPS on video."""
    print(f"[Eval] Running FPS benchmark ({max_frames} frames)...")
    detector = DroneDetector(
        weights=weights,
        conf_thresh=conf,
        iou_thresh=0.45,
        input_size=imgsz,
    )

    cap = cv2.VideoCapture(input_path)
    t_total  = 0.0
    n_frames = 0
    total_dets = 0

    while n_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0   = time.perf_counter()
        dets = detector.detect(frame)
        t_total  += time.perf_counter() - t0
        total_dets += len(dets)
        n_frames   += 1

    cap.release()

    return {
        "fps":      n_frames / t_total if t_total > 0 else 0,
        "avg_dets": total_dets / n_frames if n_frames > 0 else 0,
        "frames":   n_frames,
    }


def main():
    args = parse_args()

    val_metrics = run_official_val(args.weights, args.data, args.imgsz)

    fps_metrics = run_fps_benchmark(
        args.weights, args.input, args.imgsz, args.conf, args.max_frames
    )

    import os
    size_mb = os.path.getsize(args.weights) / 1e6

    print(f"\n{'='*55}")
    print(f"EVALUATION REPORT")
    print(f"  Weights  : {args.weights}")
    print(f"  Size     : {size_mb:.1f} MB")
    print(f"{'='*55}")
    print(f"  mAP@0.5       : {val_metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95  : {val_metrics['mAP50_95']:.4f}")
    print(f"  Precision     : {val_metrics['precision']:.4f}")
    print(f"  Recall        : {val_metrics['recall']:.4f}")
    print(f"  FPS (video)   : {fps_metrics['fps']:.1f}")
    print(f"  Avg dets/frame: {fps_metrics['avg_dets']:.1f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()