"""
Global Motion Compensator (GMC)
--------------------------------
Estimates the camera's homography between consecutive frames
using ORB keypoint matching on the background (regions not
covered by detections). Warps Kalman-predicted track positions
so ByteTrack's IoU matching works correctly despite drone movement.

Why this matters:
  Without GMC, a drone panning 50px rightward makes every predicted
  box land 50px to the left of the real person. ByteTrack sees
  zero IoU → drops the track → assigns a new ID. GMC fixes this.
"""

import cv2
import numpy as np


class GlobalMotionCompensator:
    def __init__(self, n_features: int = 500, ransac_thresh: float = 3.0):
        self.orb           = cv2.ORB_create(nfeatures=n_features)
        self.matcher       = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ransac_thresh = ransac_thresh
        self._prev_gray    = None
        self._prev_kp      = None
        self._prev_des     = None

    def update(
        self,
        frame_gray: np.ndarray,
        detection_boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute homography H from previous frame to current frame.

        Returns H (3×3) or identity if estimation fails.
        Always call this BEFORE passing detections to ByteTrack.
        """
        H = np.eye(3, dtype=np.float32)

        kp, des = self.orb.detectAndCompute(frame_gray, None)

        if (
            self._prev_gray is None
            or des is None
            or self._prev_des is None
            or len(kp) < 8
        ):
            self._store(frame_gray, kp, des)
            return H

        matches = self.matcher.match(self._prev_des, des)
        if len(matches) < 8:
            self._store(frame_gray, kp, des)
            return H

        src_pts = np.float32(
            [self._prev_kp[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        mask = self._background_mask(src_pts, detection_boxes)
        src_bg = src_pts[mask]
        dst_bg = dst_pts[mask]

        if len(src_bg) < 8:
            self._store(frame_gray, kp, des)
            return H

        H, inlier_mask = cv2.findHomography(
            src_bg, dst_bg,
            cv2.RANSAC,
            self.ransac_thresh,
        )

        if H is None:
            H = np.eye(3, dtype=np.float32)

        self._store(frame_gray, kp, des)
        return H.astype(np.float32)

    def warp_boxes(self, boxes: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warp (N,4) x1y1x2y2 boxes through homography H.
        Used to correct Kalman-predicted positions before IoU matching.
        """
        if len(boxes) == 0:
            return boxes

        corners = np.array([
            [boxes[:, 0], boxes[:, 1]],
            [boxes[:, 2], boxes[:, 3]],
        ], dtype=np.float32)
        warped = []
        for i in range(len(boxes)):
            tl = np.array([[[boxes[i, 0], boxes[i, 1]]]], dtype=np.float32)
            br = np.array([[[boxes[i, 2], boxes[i, 3]]]], dtype=np.float32)
            tl_w = cv2.perspectiveTransform(tl, H)[0][0]
            br_w = cv2.perspectiveTransform(br, H)[0][0]
            warped.append([tl_w[0], tl_w[1], br_w[0], br_w[1]])

        return np.array(warped, dtype=np.float32)

    def _background_mask(
        self, pts: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        """
        Return boolean mask: True if point is NOT inside any detection box.
        pts shape: (N, 1, 2)
        boxes shape: (M, 4) x1y1x2y2
        """
        pts_2d = pts.reshape(-1, 2)
        mask   = np.ones(len(pts_2d), dtype=bool)

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            inside = (
                (pts_2d[:, 0] >= x1) & (pts_2d[:, 0] <= x2) &
                (pts_2d[:, 1] >= y1) & (pts_2d[:, 1] <= y2)
            )
            mask &= ~inside

        return mask

    def _store(self, gray, kp, des):
        self._prev_gray = gray
        self._prev_kp   = kp
        self._prev_des  = des