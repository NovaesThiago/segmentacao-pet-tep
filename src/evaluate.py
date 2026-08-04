"""Avalia um checkpoint treinado (métricas de máscara: mAP, precisão, recall).

Uso:
    python -m src.evaluate --weights outputs/runs/train/weights/best.pt --split test
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia o modelo YOLOv8-seg.")
    parser.add_argument("--weights", default="outputs/runs/train/weights/best.pt")
    parser.add_argument("--data", default="data/oxford_pet_yolo/data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split)

    print(f"\nResultados no split '{args.split}':")
    print("\nBounding boxes")
    print(f"  Precisão:  {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")

    print("\nMáscaras de segmentação")
    print(f"  Precisão:  {metrics.seg.mp:.4f}")
    print(f"  Recall:    {metrics.seg.mr:.4f}")
    print(f"  mAP50:     {metrics.seg.map50:.4f}")
    print(f"  mAP50-95:  {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
