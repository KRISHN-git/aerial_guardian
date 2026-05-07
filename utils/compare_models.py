"""
Side-by-side comparison of baseline vs fine-tuned model.
Produces the table that goes in your report.

Usage: python utils/compare_models.py
"""

import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from models.detector import DroneDetector


MODELS = {
    "YOLOv8s (COCO baseline)":  "yolov8s.pt",
    "YOLOv8s (VisDrone FT)":    "weights/finetune/drone_person_v1/weights/best.pt",
}

VIDEO   = "data/visdrone/seq_01.mp4"
IMGSZ   = 640
CONF    = 0.15
FRAMES  = 200


def measure_fps_and_dets(weights: str) -> dict:
    detector = DroneDetector(
        weights=weights,
        conf_thresh=CONF,
        iou_thresh=0.45,
        input_size=IMGSZ,
    )
    cap      = cv2.VideoCapture(VIDEO)
    t_total  = 0.0
    n        = 0
    total_d  = 0

    while n < FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        d  = detector.detect(frame)
        t_total += time.perf_counter() - t0
        total_d += len(d)
        n       += 1

    cap.release()
    return {
        "fps":      n / t_total if t_total > 0 else 0,
        "avg_dets": total_d / n if n > 0 else 0,
        "size_mb":  os.path.getsize(weights) / 1e6,
    }


def measure_map(weights: str, data: str) -> dict:
    model   = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=IMGSZ,
        conf=0.001,
        iou=0.6,
        verbose=False,
    )
    return {
        "mAP50":    float(metrics.box.map50),
        "recall":   float(metrics.box.mr),
        "precision":float(metrics.box.mp),
    }


def main():
    print("\nRunning model comparison — this takes a few minutes...\n")

    rows = []
    for name, weights in MODELS.items():
        if not os.path.exists(weights):
            print(f"SKIP {name} — {weights} not found")
            continue

        print(f"Evaluating: {name}")
        fps_d  = measure_fps_and_dets(weights)
        map_d  = measure_map(weights, "data/drone_person.yaml")

        rows.append({
            "name":      name,
            "size_mb":   fps_d["size_mb"],
            "fps":       fps_d["fps"],
            "avg_dets":  fps_d["avg_dets"],
            "mAP50":     map_d["mAP50"],
            "recall":    map_d["recall"],
            "precision": map_d["precision"],
        })

    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON — VisDrone Person Detection")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'MB':>5} {'FPS':>6} {'Dets':>6} "
          f"{'mAP50':>7} {'Recall':>7} {'Prec':>7}")
    print(f"{'-'*80}")
    for r in rows:
        print(f"{r['name']:<30} {r['size_mb']:>5.1f} {r['fps']:>6.1f} "
              f"{r['avg_dets']:>6.1f} {r['mAP50']:>7.4f} "
              f"{r['recall']:>7.4f} {r['precision']:>7.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()