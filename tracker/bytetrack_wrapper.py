import numpy as np
import torch
from collections import defaultdict
from bytetracker import BYTETracker


class TrackState:
    """Holds per-ID trajectory history."""
    def __init__(self, max_tail: int = 30):
        self.max_tail = max_tail
        self.tails: dict[int, list] = defaultdict(list)

    def update(self, track_id: int, cx: float, cy: float):
        tail = self.tails[track_id]
        tail.append((cx, cy))
        if len(tail) > self.max_tail:
            tail.pop(0)

    def get_tail(self, track_id: int) -> list:
        return self.tails.get(track_id, [])

    def prune(self, active_ids: set):
        dead = [tid for tid in self.tails if tid not in active_ids]
        for tid in dead:
            del self.tails[tid]


class DroneTracker:
    """
    ByteTrack wrapper with trajectory tracking.

    Handles BOTH:
      - object output (t.tlbr, t.track_id)
      - numpy output ([x1,y1,x2,y2,score,id])

    Input detections:
      numpy array (N,5) -> [x1,y1,x2,y2,conf]
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        match_thresh: float = 0.7,
        track_buffer: int = 60,
        frame_rate: int = 30,
        tail_length: int = 30,
    ):
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
        )
        self.tail_state = TrackState(max_tail=tail_length)

    def update(
        self,
        detections: np.ndarray,
        frame_shape: tuple,
    ) -> list:

        if len(detections) == 0:
            return []

        img_size = (frame_shape[0], frame_shape[1])

        if detections.shape[1] == 5:
            class_col = np.zeros((len(detections), 1), dtype=np.float32)
            detections = np.hstack([detections, class_col])

        detections_tensor = torch.from_numpy(detections.astype(np.float32))

        online_targets = self.tracker.update(detections_tensor, img_size)

        results = []
        active_ids = set()

        for t in online_targets:

            if isinstance(t, np.ndarray):
                if len(t) < 6:
                    continue

                x1, y1, x2, y2 = t[:4]

                a, b = t[4], t[5]

                if float(a).is_integer():
                    tid, score = int(a), float(b)
                elif float(b).is_integer():
                    score, tid = float(a), int(b)
                else:
                    score, tid = float(a), int(b)

            else:
                x1, y1, x2, y2 = t.tlbr
                tid = int(t.track_id)
                score = float(t.score)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            self.tail_state.update(tid, cx, cy)
            active_ids.add(tid)

            results.append({
                'id': tid,
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(score),
                'tail': self.tail_state.get_tail(tid),
            })

        self.tail_state.prune(active_ids)
        return results