# Aerial Guardian — Drone Person Tracking Pipeline

Lightweight (<300MB) multi-object tracking pipeline for detecting
and tracking persons from moving drone platforms.

## Architecture Overview

### Detection: YOLOv8-S (Fine-tuned)
- Base: YOLOv8-Small — anchor-free, 22.6MB pretrained
- Fine-tuned on VisDrone DET dataset (5,667 images, 104,279 person boxes)
- Single class: person (VisDrone pedestrian + people merged)
- Final weights: 59.5MB — well within 300MB constraint

**Why YOLOv8-S:** Anchor-free head handles arbitrary small box sizes
without requiring tuned priors. C2f backbone balances accuracy and speed.
At 22.6MB base size it leaves ample room for our custom components.

### Small Object Detection: Adaptive Tiling
Our novel contribution on top of standard inference:
- Optical flow magnitude computed per frame (Farneback dense flow)
- Three tiling levels selected dynamically:
  * flow < 2.0px/frame → 4 tiles (stable scene)
  * flow 2-6px/frame  → 9 tiles (moderate motion)
  * flow > 6px/frame  → 16 tiles (high drone motion)
- Tile detections merged with custom NMS
- Recovers persons as small as 8×8 pixels

### Tracking: ByteTrack
- No ReID model required — saves 50-100MB vs DeepSORT
- Two-stage association keeps low-confidence detections
- Kalman filter predicts position between frames
- track_buffer=30: 1 second of ID persistence at 30FPS

### Ego-Motion Compensation: ORB Homography (GMC)
Our implementation (not BoT-SORT):
- ORB features extracted from background regions only
- Detection boxes masked before keypoint matching
- RANSAC homography estimated between consecutive frames
- Corrects Kalman predictions before ByteTrack association
- Prevents ID switching during drone panning/tilting

## Performance Results

Hardware: Intel i5-12500H, CPU only

| Config | FPS | Avg Tracks | Notes |
|---|---|---|---|
| Baseline 640px | 6.0 | 0.5 | COCO weights |
| Baseline 960px | 3.3 | 1.5 | COCO weights |
| Fine-tuned 960px | 3.4 | 10.0 | **Selected config** |
| Fine-tuned + Tiled | 0.7 | 9.0 | GPU recommended |
| Fine-tuned + GMC | 0.5 | 9.0 | Full pipeline |
| ONNX Runtime | ~5.0 | 10.0 | 37.9MB, 1.5x speedup |

## Model Sizes

| Component | Size |
|---|---|
| YOLOv8s fine-tuned | 59.5 MB |
| ONNX export | 37.9 MB |
| ByteTrack | <1 MB |
| Total pipeline | **<61 MB** / 300MB limit |

## Fine-tuning Results

| Metric | COCO Baseline | VisDrone FT (10ep) |
|---|---|---|
| mAP@0.5 | 0.2895 | 0.4327 (+49%) |
| Recall | 0.2511 | 0.4066 (+62%) |
| Precision | 0.5077 | 0.5802 (+14%) |
| FPS | 4.3 | 5.7 (+33%) |

## Setup

```bash
# Clone repo
git clone <your-repo-url>
cd aerial_guardian

# Create environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run Pipeline

```bash
# Standard inference
python infer_track.py \
    --input  your_video.mp4 \
    --output outputs/result.mp4 \
    --weights weights/drone_person_best.pt \
    --imgsz  960

# With adaptive tiling
python infer_track.py \
    --input  your_video.mp4 \
    --output outputs/result_tiled.mp4 \
    --weights weights/drone_person_best.pt \
    --imgsz  640 \
    --tiled

# Full pipeline (tiling + GMC)
python infer_track.py \
    --input  your_video.mp4 \
    --output outputs/result_full.mp4 \
    --weights weights/drone_person_best.pt \
    --imgsz  640 \
    --tiled \
    --use-gmc
```

## Edge Deployment

See `docs/edge_deployment.md` for full Jetson deployment guide.

```bash
# Export to ONNX
python -c "
from ultralytics import YOLO
model = YOLO('weights/drone_person_best.pt')
model.export(format='onnx', simplify=False)
"

# Run with ONNX Runtime
python infer_track_onnx.py --input video.mp4 --output result.mp4
```

## Repo Structure

```
AerialGuardian/
├── data/
│   ├── drone_person.yaml
│   └── yolo_dataset/
├── models/
│   ├── detector.py
│   └── onnx_detector.py
├── tracker/
│   ├── bytetrack_wrapper.py
│   └── gmc.py
├── utils/
│   ├── tiler.py
│   ├── visualiser.py
│   ├── annotation_parser.py
│   └── convert_annotations.py
├── docs/
│   └── edge_deployment.md
├── weights/
│   ├── drone_person_best.pt
│   └── drone_person_best.onnx
├── outputs/
│   └── FINAL_aerial_guardian.mp4
├── infer_track.py
├── train.py
├── evaluate.py
├── benchmark.py
├── benchmark_onnx.py
├── requirements.txt
└── README.md
```