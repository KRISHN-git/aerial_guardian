"""
Compares PyTorch vs ONNX Runtime inference speed.
This is your edge optimization evidence.
"""

import cv2
import time
import numpy as np
from models.detector      import DroneDetector
from models.onnx_detector import ONNXDroneDetector

VIDEO      = "data/visdrone/seq_01.mp4"
MAX_FRAMES = 150
IMGSZ      = 640
CONF       = 0.15


def benchmark(detector, label: str) -> dict:
    cap     = cv2.VideoCapture(VIDEO)
    t_total = 0.0
    n       = 0
    total_d = 0

    while n < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        d  = detector.detect(frame)
        t_total += time.perf_counter() - t0
        total_d += len(d)
        n       += 1

    cap.release()
    fps = n / t_total if t_total > 0 else 0

    print(f"  {label:<30} {fps:>6.1f} FPS | avg dets: {total_d/n:.1f}")
    return {"label": label, "fps": fps, "avg_dets": total_d / n}


def main():
    print(f"\n[Benchmark] PyTorch vs ONNX — {MAX_FRAMES} frames\n")

    pt_det   = DroneDetector(
        weights="weights/drone_person_best.pt",
        conf_thresh=CONF,
        input_size=IMGSZ,
    )
    onnx_det = ONNXDroneDetector(
        onnx_path="weights/drone_person_best.onnx",
        conf_thresh=CONF,
        input_size=IMGSZ,
    )

    results = []
    results.append(benchmark(pt_det,   "PyTorch CPU (best.pt)"))
    results.append(benchmark(onnx_det, "ONNX Runtime CPU"))

    speedup = results[1]["fps"] / results[0]["fps"]

    print(f"\n{'='*55}")
    print(f"ONNX speedup over PyTorch: {speedup:.2f}x")
    print(f"Model size: PyTorch={59.5:.1f}MB  ONNX={37.9:.1f}MB")
    print(f"Size reduction: {(1 - 37.9/59.5)*100:.1f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()