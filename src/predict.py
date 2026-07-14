"""Roda inferência em algumas imagens do split de teste e salva as imagens
com a máscara de segmentação prevista sobreposta.

Uso:
    python -m src.predict --weights outputs/runs/train/weights/best.pt --num-samples 6
"""

import argparse
import random
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Gera amostras visuais de predição.")
    parser.add_argument("--weights", default="outputs/runs/train/weights/best.pt")
    parser.add_argument("--source", default="data/oxford_pet_yolo/images/test")
    parser.add_argument("--output-dir", default="outputs/samples")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    all_images = sorted(Path(args.source).glob("*.jpg"))
    sample_images = random.sample(all_images, min(args.num_samples, len(all_images)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(source=[str(p) for p in sample_images], conf=args.conf, save=False)

    for img_path, result in zip(sample_images, results):
        out_path = output_dir / f"{img_path.stem}_pred.png"
        result.save(filename=str(out_path))
        print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
