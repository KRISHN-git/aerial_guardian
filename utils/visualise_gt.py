import cv2
import numpy as np
from pathlib import Path
from annotation_parser import load_annotation

def visualise_sequence(seq_name: str, n_frames: int = 60):
    seq_dir  = Path(f'data/visdrone/sequences/{seq_name}')
    ann_path = Path(f'data/visdrone/annotations/{seq_name}.txt')
    
    df = load_annotation(ann_path)
    frames = sorted(seq_dir.glob('*.jpg'))[:n_frames]
    
    out = cv2.VideoWriter(
        f'outputs/gt_{seq_name}.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), 10,
        (cv2.imread(str(frames[0])).shape[1],
         cv2.imread(str(frames[0])).shape[0])
    )
    
    for i, fpath in enumerate(frames):
        img = cv2.imread(str(fpath))
        frame_anns = df[df['frame'] == i + 1]
        
        for _, row in frame_anns.iterrows():
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(img, f"ID:{int(row['target_id'])}",
                        (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
        
        cv2.putText(img, f"Frame {i+1} | Persons: {len(frame_anns)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        out.write(img)
    
    out.release()
    print(f"Saved: outputs/gt_{seq_name}.mp4")

if __name__ == '__main__':
    visualise_sequence('uav0000086_00000_v', n_frames=80)