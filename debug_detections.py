import cv2
import numpy as np
from models.detector import DroneDetector

detector = DroneDetector(
    weights="yolov8s.pt",
    conf_thresh=0.25,
    iou_thresh=0.45,
    input_size=640,
)

cap = cv2.VideoCapture("data/visdrone/seq_01.mp4")
found_frame = None

for i in range(50):
    ret, frame = cap.read()
    if not ret:
        break
    dets = detector.detect(frame)
    print(f"Frame {i+1}: {len(dets)} detections")
    if len(dets) > 0:
        print(f"  Sample det: {dets[0]}")
        found_frame = frame.copy()
        break

cap.release()

if found_frame is not None:
    print("\nDetections found — tracker input issue")
else:
    print("\nZero detections on all frames — detector issue")