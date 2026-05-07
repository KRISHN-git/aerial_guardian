RESULTS = [
    # config                        FPS    Avg Tracks  notes
    ("Baseline 640px",              6.0,   0.5,   "YOLOv8s COCO, no tiling, CPU"),
    ("Baseline 960px",              3.3,   1.5,   "YOLOv8s COCO, no tiling, CPU"),
    ("Tiled 640px (COCO)",          0.6,   7.0,   "COCO weights + adaptive tiling"),
    ("Tiled 960px (COCO)",          0.3,   10.0,  "COCO weights + tiling, too slow"),
    ("Fine-tuned 960px",            3.4,   10.0,  "VisDrone FT, best FPS/recall balance"),
    ("Fine-tuned + Tiled 640px",    0.7,   9.0,   "FT + adaptive tiling, GPU needed"),
    ("Fine-tuned + Tiled + GMC",    0.5,   9.0,   "Full pipeline, GPU needed for RT"),
    ("ONNX export",                 None,  None,  "Day 8 — edge deployment"),
]

if __name__ == "__main__":
    print(f"\n{'='*75}")
    print(f"AERIAL GUARDIAN — BENCHMARK RESULTS")
    print(f"Hardware: Intel i5-12500H, CPU only")
    print(f"{'='*75}")
    print(f"{'Config':<30} {'FPS':>6} {'Tracks':>8}  Notes")
    print(f"{'-'*75}")
    for cfg, fps, tracks, note in RESULTS:
        fps_str    = f"{fps:.1f}"    if fps    is not None else "—"
        track_str  = f"{tracks:.1f}" if tracks is not None else "—"
        print(f"{cfg:<30} {fps_str:>6} {track_str:>8}  {note}")
    print(f"{'='*75}")
    print(f"\nSelected config : Fine-tuned 960px (3.4 FPS, ~10 avg tracks)")
    print(f"Edge deployment : FP16/TensorRT on Jetson → estimated 15-25 FPS")
    print()