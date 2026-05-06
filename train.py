import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  default="yolov8s.pt")
    p.add_argument("--data",     default="data/drone_person.yaml")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=8)
    p.add_argument("--device",   default="cpu")
    p.add_argument("--project",  default="weights/finetune")
    p.add_argument("--name",     default="drone_person_v1")
    p.add_argument("--resume",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    print(f"\n[Train] Starting fine-tune")
    print(f"  Weights  : {args.weights}")
    print(f"  Data     : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Img size : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        degrees=15,
        translate=0.2,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=15,
        save_period=10,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        cache=False,
        workers=2,
        verbose=True,
    )

    print(f"\n[Train] Complete.")
    print(f"  Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()