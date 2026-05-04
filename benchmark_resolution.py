"""
Runs the pipeline at multiple resolutions and prints a comparison table.
Usage: python benchmark_resolution.py --input data/visdrone/seq_01.mp4
"""

import cv2
import time
import argparse
import numpy as np
from models.detector import DroneDetector
from tracker.bytetrack_wrapper import DroneTracker


def run_resolution_test(
    input_path: str,
    weights: str,
    imgsz: int,
    max_frames: int = 150,
    conf: float = 0.15,
) -> dict:
    """Run pipeline at given resolution, return metrics dict."""

    detector = DroneDetector(
        weights=weights,
        conf_thresh=conf,
        iou_thresh=0.45,
        input_size=imgsz,
    )
    tracker = DroneTracker(
        track_thresh=0.3,
        match_thresh=0.8,
        track_buffer=30,
        tail_length=30,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {input_path}")

    total_det   = 0
    total_track = 0
    n_frames    = 0
    t_det       = 0.0
    t_track     = 0.0

    while n_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t_det += time.perf_counter() - t0

        t1 = time.perf_counter()
        tracks = tracker.update(dets, frame.shape[:2])
        t_track += time.perf_counter() - t1

        total_det   += len(dets)
        total_track += len(tracks)
        n_frames    += 1

    cap.release()

    fps_det  = n_frames / t_det   if t_det   > 0 else 0
    fps_full = n_frames / (t_det + t_track) if (t_det + t_track) > 0 else 0

    return {
        "imgsz":        imgsz,
        "fps_det":      fps_det,
        "fps_full":     fps_full,
        "avg_dets":     total_det   / n_frames,
        "avg_tracks":   total_track / n_frames,
        "frames":       n_frames,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   required=True)
    p.add_argument("--weights", default="yolov8s.pt")
    p.add_argument("--frames",  type=int, default=150,
                   help="Frames to test per resolution (150 is enough for stable FPS)")
    args = p.parse_args()

    resolutions = [640, 960, 1280]
    results     = []

    for sz in resolutions:
        print(f"\n[Benchmark] Testing {sz}px — {args.frames} frames...")
        r = run_resolution_test(
            args.input,
            args.weights,
            sz,
            max_frames=args.frames,
        )
        results.append(r)
        print(f"  Done — {r['fps_full']:.1f} FPS | "
              f"avg dets: {r['avg_dets']:.1f} | "
              f"avg tracks: {r['avg_tracks']:.1f}")

    # Print comparison table
    print(f"\n{'='*65}")
    print(f"RESOLUTION BENCHMARK — {args.weights}")
    print(f"{'='*65}")
    print(f"{'Resolution':<14} {'Det FPS':>8} {'Full FPS':>9} "
          f"{'Avg Dets':>9} {'Avg Tracks':>11}")
    print(f"{'-'*65}")
    for r in results:
        print(f"{str(r['imgsz'])+'px':<14} "
              f"{r['fps_det']:>8.1f} "
              f"{r['fps_full']:>9.1f} "
              f"{r['avg_dets']:>9.1f} "
              f"{r['avg_tracks']:>11.1f}")
    print(f"{'='*65}")

    viable = [r for r in results if r['fps_full'] >= 2.0]
    if viable:
        best = max(viable, key=lambda r: r['avg_tracks'])
        print(f"\nRecommended : {best['imgsz']}px")
        print(f"Reasoning   : {best['avg_tracks']:.1f} avg tracks at {best['fps_full']:.1f} FPS")
        print(f"              Tiling (Day 4) will recover FPS — prioritise recall now")
    print()


if __name__ == "__main__":
    main()