import cv2
from pathlib import Path

def sequence_to_video(seq_dir: str, out_path: str, fps: float = 30.0):
    frames = sorted(Path(seq_dir).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No JPEGs in {seq_dir}")
    sample = cv2.imread(str(frames[0]))
    H, W   = sample.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()
    print(f"Saved {len(frames)} frames → {out_path}")

if __name__ == "__main__":
    sequence_to_video(
        "data/visdrone/sequences/uav0000182_00000_v",
        "data/visdrone/seq_01.mp4"
    )