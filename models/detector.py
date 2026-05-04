import cv2
import os
import torch
import numpy as np
from typing import Tuple
from ultralytics import YOLO

PERSON_CLASS_ID = 0


class DroneDetector:
    """
    Thin wrapper around YOLOv8 for drone person detection.
    Keeping this as a class means we can swap the underlying
    model (standard vs P2-head vs fine-tuned) without touching
    the pipeline code.
    """

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.45,
        input_size: int = 960,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf   = conf_thresh
        self.iou    = iou_thresh
        self.imgsz  = input_size

        print(f"[DroneDetector] Loading {weights} on {self.device}")
        self.model  = YOLO(weights)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single BGR frame.

        Returns
        -------
        np.ndarray of shape (N, 5) — [x1, y1, x2, y2, confidence]
        Only person-class detections are returned.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().reshape(-1, 1)

        return np.hstack([xyxy, confs]).astype(np.float32)

    def detect_tiled(
        self,
        frame: np.ndarray,
        tiler,
    ) -> Tuple[np.ndarray, dict]:
        """
        Run tiled inference using adaptive tiling strategy.

        Returns
        -------
        detections : (N, 5) [x1, y1, x2, y2, conf]
        info       : dict with tiling metadata for logging
        """
        from utils.tiler import slice_frame, merge_detections

        frame_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cfg           = tiler.get_config(frame_gray)
        tiles, coords = slice_frame(frame, cfg)

        tile_dets = []
        for tile in tiles:
            dets = self.detect(tile)
            tile_dets.append(dets)

        merged = merge_detections(tile_dets, coords, iou_thresh=self.iou)

        info = {
            "n_tiles":   len(tiles),
            "flow_mag":  tiler.last_flow_magnitude,
            "strategy":  tiler.last_config_name,
            "raw_dets":  sum(len(d) for d in tile_dets),
            "after_nms": len(merged),
        }

        return merged, info

    @property
    def model_size_mb(self) -> float:
        """Return approximate model size for the report."""
        try:
            path = self.model.ckpt_path
            return os.path.getsize(path) / 1e6
        except Exception:
            return -1.0