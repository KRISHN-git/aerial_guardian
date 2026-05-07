# Edge Deployment Guide — Aerial Guardian

## Target Hardware: NVIDIA Jetson Orin Nano (8GB)

### Deployment Path: PyTorch → ONNX → TensorRT

### Step 1 — Export ONNX (already done)
```bash
python -c "
from ultralytics import YOLO
model = YOLO('weights/drone_person_best.pt')
model.export(format='onnx', opset=11, simplify=False)
"
```

### Step 2 — Convert to TensorRT on Jetson
```bash
# Run this ON the Jetson device
trtexec \
    --onnx=drone_person_best.onnx \
    --saveEngine=drone_person_best.engine \
    --fp16 \
    --workspace=1024
```

### Step 3 — Run with TensorRT engine
```bash
# Ultralytics supports TensorRT directly
yolo predict \
    model=drone_person_best.engine \
    source=video.mp4 \
    imgsz=640
```

## Expected Performance on Jetson Orin Nano

| Config | CPU (i5-12500H) | Jetson FP32 | Jetson FP16 |
|---|---|---|---|
| YOLOv8s PyTorch | 3.4 FPS | ~12 FPS | ~22 FPS |
| YOLOv8s ONNX | ~5 FPS | ~18 FPS | ~30 FPS |
| YOLOv8s TensorRT FP16 | N/A | ~25 FPS | ~40 FPS |
| + Adaptive Tiling | 0.7 FPS | ~8 FPS | ~15 FPS |

## Optimisation Strategies for Edge

### 1. FP16 Quantisation
Halves memory bandwidth, minimal accuracy loss on detection tasks.
mAP drop typically < 1% on drone data.

### 2. Input Resolution Reduction
640px → 416px reduces compute by ~57% with ~8% mAP drop.
Acceptable for real-time tracking at low altitude.

### 3. Adaptive Tiling Toggle
Disable tiling when drone is stable (flow_mag < 2.0).
Enable only during active camera motion.
Our AdaptiveTiler already implements this automatically.

### 4. Tracker Optimisation
ByteTrack runs purely on CPU with numpy — no GPU needed.
Kalman filter operations are negligible (<0.5ms per frame).

### 5. Model Size Summary
| Component | Size |
|---|---|
| YOLOv8s fine-tuned (.pt) | 59.5 MB |
| YOLOv8s ONNX | 37.9 MB |
| ByteTrack (pure Python) | <1 MB |
| GMC (OpenCV ORB) | 0 MB |
| **Total pipeline** | **<61 MB** |
| **Limit** | **300 MB** |
| **Headroom** | **239 MB** |