import pandas as pd
import numpy as np
from pathlib import Path

COLS = [
    'frame', 'target_id', 'x', 'y', 'w', 'h',
    'score', 'category', 'truncation', 'occlusion'
]

PERSON_CLASSES = {1, 2}

def load_annotation(ann_path: str) -> pd.DataFrame:
    """Load a single VisDrone annotation file, filter to person classes."""
    df = pd.read_csv(ann_path, header=None, names=COLS)
    df = df[df['category'].isin(PERSON_CLASSES)].copy()
    df['x2'] = df['x'] + df['w']
    df['y2'] = df['y'] + df['h']
    df['area'] = df['w'] * df['h']
    return df.reset_index(drop=True)

def summarise_dataset(ann_dir: str):
    """Print key statistics across all annotation files."""
    ann_dir = Path(ann_dir)
    all_dfs = []
    
    for ann_file in sorted(ann_dir.glob('*.txt')):
        df = load_annotation(ann_file)
        df['sequence'] = ann_file.stem
        all_dfs.append(df)
    
    full = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n{'='*50}")
    print(f"Total person annotations : {len(full):,}")
    print(f"Sequences                : {full['sequence'].nunique()}")
    print(f"Unique track IDs         : {full['target_id'].nunique()}")
    print(f"\n--- Bounding box size distribution ---")
    print(f"Median width  : {full['w'].median():.1f} px")
    print(f"Median height : {full['h'].median():.1f} px")
    print(f"Mean area     : {full['area'].mean():.1f} px²")
    
    tiny = full[(full['w'] < 32) & (full['h'] < 32)]
    pct = 100 * len(tiny) / len(full)
    print(f"\nPersons smaller than 32×32 px : {len(tiny):,} ({pct:.1f}%)")
    
    small = full[(full['w'] < 64) & (full['h'] < 64)]
    pct2 = 100 * len(small) / len(full)
    print(f"Persons smaller than 64×64 px : {len(small):,} ({pct2:.1f}%)")
    
    print(f"\n--- Occlusion levels (0=none, 1=partial, 2=heavy) ---")
    print(full['occlusion'].value_counts().to_string())
    print(f"{'='*50}\n")
    
    return full

if __name__ == '__main__':
    df = summarise_dataset('data/visdrone/annotations')