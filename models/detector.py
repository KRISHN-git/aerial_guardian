import torch
import numpy as np
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
        conf_thresh: float = 0.10,
        iou_thresh: float = 0.45,
        input_size: int = 640,
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
        Run detection on a single BGR frame (as returned by cv2.imread).

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

    @property
    def model_size_mb(self) -> float:
        """Return approximate model size for the report."""
        import os
        try:
            path = self.model.ckpt_path
            return os.path.getsize(path) / 1e6
        except Exception:
            return -1.0