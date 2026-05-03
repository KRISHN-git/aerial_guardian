import cv2
import numpy as np

_PALETTE = [
    (255, 128, 0), (0, 200, 255), (0, 255, 128),
    (255, 0, 128), (128, 0, 255), (255, 220, 0),
    (0, 128, 255), (200, 255, 0), (255, 0, 200),
]

def _colour(track_id: int) -> tuple:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_tracks(
    frame:   np.ndarray,
    tracks:  list,
    show_conf: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes, ID labels, and fading trajectory tails.

    Tail fading: oldest points are drawn at low opacity (thin),
    newest points at full opacity (thick). This is done by drawing
    each segment with increasing alpha — achieved via cv2.addWeighted
    on a transparent overlay.
    """
    overlay = frame.copy()

    for track in tracks:
        tid   = track['id']
        x1, y1, x2, y2 = [int(v) for v in track['box']]
        tail  = track['tail']
        colour = _colour(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        label = f"ID:{tid}"
        if show_conf:
            label += f" {track['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

        if len(tail) >= 2:
            n = len(tail)
            for i in range(1, n):
                alpha     = i / n
                thickness = max(1, int(alpha * 3))
                pt1 = (int(tail[i-1][0]), int(tail[i-1][1]))
                pt2 = (int(tail[i][0]),   int(tail[i][1]))

                seg_overlay = frame.copy()
                cv2.line(seg_overlay, pt1, pt2, colour, thickness, cv2.LINE_AA)
                cv2.addWeighted(seg_overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)

    return frame


def draw_hud(
    frame:     np.ndarray,
    fps:       float,
    n_tracks:  int,
    label:     str = "BASELINE",
) -> np.ndarray:
    """Draw FPS and track count HUD in top-left corner."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (220, 56), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}",      (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, f"FPS : {fps:.1f}",     (8, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,128), 1)
    cv2.putText(frame, f"IDs : {n_tracks}",    (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
    return frame