RESULTS = [
    ("Baseline 640px",   6.0,  "YOLOv8s stock, conf=0.15, no tiling, no GMC, CPU"),
    ("Baseline 960px",   3.3,  "YOLOv8s stock, +83% tracks vs 640px"),
    ("Baseline 1280px",  1.7,  "YOLOv8s stock, too slow for real-time"),
    ("SAHI tiling",      None, "Day 4 — expect FPS recovery + more tracks"),
    ("Fine-tuned model", None, "Day 6"),
    ("+ GMC",            None, "Day 7"),
    ("FP16 ONNX",        None, "Day 8"),
]

# I have the configs, FPS and notes.