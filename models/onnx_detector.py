"""
ONNX Runtime inference wrapper for DroneDetector.
Replaces PyTorch inference with ONNX Runtime for edge deployment.

Why ONNX Runtime:
  - Runs on CPU without PyTorch installed
  - 20-40% faster than PyTorch CPU inference
  - Directly converts to TensorRT on Jetson
  - Single file deployment — no framework needed
"""

import cv2
import time
import numpy as np
import onnxruntime as ort


PERSON_CLASS_ID = 0


class ONNXDroneDetector:
    """
    ONNX Runtime version of DroneDetector.
    Drop-in replacement — same detect() interface.
    """

    def __init__(
        self,
        onnx_path: str = "weights/drone_person_best.onnx",
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.45,
        input_size: int = 640,
    ):
        self.conf      = conf_thresh
        self.iou       = iou_thresh
        self.imgsz     = input_size
        self.onnx_path = onnx_path

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        import os
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"[ONNXDetector] Loaded {onnx_path} ({size_mb:.1f} MB)")
        print(f"[ONNXDetector] Provider: {self.session.get_providers()[0]}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Same interface as DroneDetector.detect().
        Returns (N, 5) array [x1, y1, x2, y2, conf]
        """
        blob = self._preprocess(frame)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: blob}
        )[0]

        return self._postprocess(outputs, frame.shape)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, BCHW format."""
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img

    def _postprocess(
        self,
        outputs: np.ndarray,
        orig_shape: tuple,
    ) -> np.ndarray:
        """
        Convert raw ONNX output to [x1, y1, x2, y2, conf] in original coords.
        Output shape from YOLOv8: (1, 5, 8400)
          - 5 = [cx, cy, w, h, conf_person]
          - 8400 = number of anchors
        """
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz

        pred = outputs[0]
        pred = pred.T

        confs = pred[:, 4]
        mask  = confs >= self.conf
        pred  = pred[mask]

        if len(pred) == 0:
            return np.empty((0, 5), dtype=np.float32)

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - w / 2) * scale_x
        y1 = (cy - h / 2) * scale_y
        x2 = (cx + w / 2) * scale_x
        y2 = (cy + h / 2) * scale_y
        confs = pred[:, 4]

        boxes = np.stack([x1, y1, x2, y2, confs], axis=1).astype(np.float32)

        keep = self._nms(boxes, self.iou)
        return boxes[keep]

    def _nms(self, dets: np.ndarray, iou_thresh: float) -> list:
        """Simple NMS."""
        if len(dets) == 0:
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep  = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w   = np.maximum(0.0, xx2 - xx1 + 1)
            h   = np.maximum(0.0, yy2 - yy1 + 1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
            inds  = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    @property
    def model_size_mb(self) -> float:
        import os
        return os.path.getsize(self.onnx_path) / 1e6