"""Treina o YOLOv8-seg no dataset preparado por `prepare_dataset.py`.

Uso:
    python -m src.train --epochs 30 --batch 16
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Treina YOLOv8-seg.")
    parser.add_argument("--data", default="data/oxford_pet_yolo/data.yaml")
    parser.add_argument("--model", default="yolov8n-seg.pt",
                         help="Checkpoint pré-treinado usado como ponto de partida.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15,
                        help="Épocas sem melhora antes do early stopping.")
    parser.add_argument("--project", default="outputs/runs")
    parser.add_argument("--name", default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
