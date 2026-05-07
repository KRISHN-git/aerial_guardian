import os

videos = [
    'outputs/FINAL_aerial_guardian.mp4',
    'outputs/baseline_640.mp4',
    'outputs/finetuned_960.mp4',
]

for v in videos:
    exists = os.path.exists(v)
    size = os.path.getsize(v) / 1e6 if exists else 0
    print(f"{'OK' if exists else 'MISSING'} {v} ({size:.1f}MB)")