import cv2
import time
import argparse
import numpy as np
from pathlib import Path

from models.detector        import DroneDetector
from tracker.bytetrack_wrapper import DroneTracker
from tracker.gmc            import GlobalMotionCompensator
from utils.visualiser       import draw_tracks, draw_hud


def parse_args():
    p = argparse.ArgumentParser(description="Aerial Guardian — drone MOT pipeline")
    p.add_argument("--input",     required=True,       help="Input video path or image folder")
    p.add_argument("--output",    default="outputs/result.mp4")
    p.add_argument("--weights",   default="yolov8s.pt")
    p.add_argument("--conf",      type=float, default=0.25)
    p.add_argument("--iou",       type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--use-gmc",   action="store_true",
                   help="Enable global motion compensation")
    p.add_argument("--label",     default="BASELINE",
                   help="Label shown in HUD — change per experiment")
    p.add_argument("--max-frames",type=int,   default=None,
                   help="Cap frames for quick testing")
    return p.parse_args()


def open_source(input_path: str):
    """Return (cv2.VideoCapture, fps, total_frames)."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total


def open_writer(output_path: str, fps: float, frame_size: tuple):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def main():
    args = parse_args()

    detector = DroneDetector(
        weights=args.weights,
        conf_thresh=0.15,
        iou_thresh=0.45,
        input_size=args.imgsz,
    )
    tracker  = DroneTracker(
        track_thresh=0.25,
        match_thresh=0.8,
        track_buffer=30,
        tail_length=30,
    )
    gmc = GlobalMotionCompensator() if args.use_gmc else None

    print(f"[Pipeline] Model size : {detector.model_size_mb:.1f} MB")
    print(f"[Pipeline] GMC        : {'ON' if gmc else 'OFF'}")
    print(f"[Pipeline] Input size : {args.imgsz}px")

    cap, src_fps, total_frames = open_source(args.input)
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")
    H, W = first.shape[:2]

    writer = open_writer(args.output, src_fps, (W, H))

    t_det   = 0.0
    t_track = 0.0
    t_draw  = 0.0
    n_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\n[Pipeline] Processing {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and n_frames >= args.max_frames:
            break

        n_frames += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t_det += time.perf_counter() - t0

        t1 = time.perf_counter()
        if gmc is not None and len(detections) > 0:
            H_mat = gmc.update(frame_gray, detections[:, :4])
            
        tracks = tracker.update(detections, frame.shape[:2])
        t_track += time.perf_counter() - t1

        t2 = time.perf_counter()
        frame = draw_tracks(frame, tracks)

        elapsed    = t_det + t_track
        fps_so_far = n_frames / elapsed if elapsed > 0 else 0.0
        frame = draw_hud(frame, fps_so_far, len(tracks), label=args.label)
        t_draw += time.perf_counter() - t2

        writer.write(frame)

        if n_frames % 50 == 0:
            print(f"  Frame {n_frames}/{total_frames or '?'} | "
                  f"FPS {fps_so_far:.1f} | Tracks {len(tracks)}")

    cap.release()
    writer.release()

    avg_fps_det   = n_frames / t_det   if t_det   > 0 else 0
    avg_fps_full  = n_frames / (t_det + t_track) if (t_det + t_track) > 0 else 0

    print(f"\n{'='*50}")
    print(f"BENCHMARK — {args.label}")
    print(f"{'='*50}")
    print(f"Frames processed  : {n_frames}")
    print(f"Detection FPS     : {avg_fps_det:.1f}")
    print(f"Full pipeline FPS : {avg_fps_full:.1f}")
    print(f"Output saved to   : {args.output}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()