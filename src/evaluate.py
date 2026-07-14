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
    for key, value in metrics.results_dict.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
