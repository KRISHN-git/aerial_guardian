"""
Adaptive Tiled Inference
------------------------
Novel contribution: instead of tiling every frame uniformly,
we use optical flow magnitude to decide tiling aggressiveness.

High camera motion  → more tiles (drone panning, need high recall)
Stable scene        → fewer tiles (save compute, maintain FPS)

This is built ON TOP of SAHI's slicing utility — we use their
merge/NMS logic but control the tiling strategy ourselves.

Why optical flow for motion detection:
  cv2.calcOpticalFlowFarneback gives dense flow per pixel.
  We take the mean magnitude across the frame — high mean = camera moved.
  Person motion alone wouldn't raise the mean significantly
  since persons are small relative to the frame.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TileConfig:
    """Defines how to slice a frame."""
    slice_h:     int 
    slice_w:     int
    overlap:     float
    n_tiles_est: int 
    reason:      str 


TILE_CONFIGS = {
    "full":       TileConfig(960,  960,  0.0,  4,  "stable scene"),
    "standard":   TileConfig(640,  640,  0.2,  9,  "moderate motion"),
    "aggressive": TileConfig(512,  512,  0.3,  16, "high drone motion"),
}

FLOW_THRESH_LOW  = 2.0
FLOW_THRESH_HIGH = 6.0  


class AdaptiveTiler:
    """
    Computes optical flow between frames and selects
    the appropriate tiling strategy dynamically.
    """

    def __init__(self, warmup_frames: int = 5):
        self._prev_gray  = None
        self._frame_idx  = 0
        self._warmup     = warmup_frames
        self._last_flow  = 0.0
        self._last_cfg   = TILE_CONFIGS["standard"]

    def get_config(self, frame_gray: np.ndarray) -> TileConfig:
        """
        Given the current grayscale frame, return the TileConfig to use.
        During warmup frames, always use standard tiling.
        """
        self._frame_idx += 1

        if self._prev_gray is None or self._frame_idx <= self._warmup:
            self._prev_gray = frame_gray.copy()
            return TILE_CONFIGS["standard"]

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray,
            frame_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_mag  = float(np.mean(magnitude))
        self._last_flow = mean_mag

        if mean_mag < FLOW_THRESH_LOW:
            cfg = TILE_CONFIGS["full"]
        elif mean_mag < FLOW_THRESH_HIGH:
            cfg = TILE_CONFIGS["standard"]
        else:
            cfg = TILE_CONFIGS["aggressive"]

        self._last_cfg  = cfg
        self._prev_gray = frame_gray.copy()
        return cfg

    @property
    def last_flow_magnitude(self) -> float:
        return self._last_flow

    @property
    def last_config_name(self) -> str:
        return self._last_cfg.reason


def slice_frame(
    frame: np.ndarray,
    cfg: TileConfig,
) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    """
    Slice a frame into overlapping tiles.

    Returns:
        tiles     : list of BGR image crops
        tile_coords: list of (x1, y1, x2, y2) in original frame coords
    """
    H, W  = frame.shape[:2]
    sh, sw = cfg.slice_h, cfg.slice_w
    oh    = int(sh * cfg.overlap)
    ow    = int(sw * cfg.overlap)
    step_h = sh - oh
    step_w = sw - ow

    tiles       = []
    tile_coords = []

    y = 0
    while y < H:
        x = 0
        y2 = min(y + sh, H)
        while x < W:
            x2 = min(x + sw, W)
            tile = frame[y:y2, x:x2]
            tiles.append(tile)
            tile_coords.append((x, y, x2, y2))
            x += step_w
            if x2 == W:
                break
        y += step_h
        if y2 == H:
            break

    return tiles, tile_coords


def merge_detections(
    tile_dets:   List[np.ndarray],
    tile_coords: List[Tuple[int,int,int,int]],
    iou_thresh:  float = 0.5,
) -> np.ndarray:
    """
    Map per-tile detections back to full-frame coordinates,
    then apply NMS to remove duplicates at tile boundaries.

    Args:
        tile_dets   : list of (N,5) arrays [x1,y1,x2,y2,conf] per tile
        tile_coords : list of (x1,y1,x2,y2) offsets for each tile
        iou_thresh  : NMS IoU threshold

    Returns:
        np.ndarray of shape (M, 5) in full-frame coords
    """
    all_dets = []

    for dets, (tx1, ty1, tx2, ty2) in zip(tile_dets, tile_coords):
        if len(dets) == 0:
            continue
        shifted = dets.copy()
        shifted[:, 0] += tx1 
        shifted[:, 1] += ty1
        shifted[:, 2] += tx1
        shifted[:, 3] += ty1
        all_dets.append(shifted)

    if not all_dets:
        return np.empty((0, 5), dtype=np.float32)

    all_dets = np.vstack(all_dets)

    keep = _nms(all_dets, iou_thresh)
    return all_dets[keep]


def _nms(dets: np.ndarray, iou_thresh: float) -> List[int]:
    """
    Simple CPU NMS.
    dets: (N, 5) [x1, y1, x2, y2, conf]
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou   = inter / (areas[i] + areas[order[1:]] - inter)

        inds  = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep