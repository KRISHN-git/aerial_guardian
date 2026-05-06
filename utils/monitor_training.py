"""
Reads results.csv from Ultralytics training and prints
a clean progress summary. Run while training is active.

Usage: python utils/monitor_training.py
"""

import time
import pandas as pd
from pathlib import Path


def monitor(results_csv: str, refresh_sec: int = 30):
    path = Path(results_csv)
    print(f"Monitoring: {path}")
    print("Refreshes every 30 seconds. Ctrl+C to stop.\n")

    while True:
        try:
            if not path.exists():
                print("Waiting for training to start...")
                time.sleep(refresh_sec)
                continue

            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()

            if len(df) == 0:
                time.sleep(refresh_sec)
                continue

            latest = df.iloc[-1]
            epoch  = int(latest.get("epoch", len(df)))

            print(f"\n[Epoch {epoch}]")
            for col in df.columns:
                if any(k in col for k in ["loss", "mAP", "precision", "recall"]):
                    try:
                        print(f"  {col.strip():30s}: {float(latest[col]):.4f}")
                    except Exception:
                        pass

        except Exception as e:
            print(f"Read error: {e}")

        time.sleep(refresh_sec)


if __name__ == "__main__":
    monitor("weights/finetune/drone_person_v1/results.csv")