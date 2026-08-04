"""Desenha polígonos YOLO-seg sobre imagens para inspeção das anotações.

Uso:
    python -m src.validate_annotations --num-samples 6
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Valida visualmente anotações YOLO-seg.")
    parser.add_argument("--images", default="data/oxford_pet_yolo/images/train")
    parser.add_argument("--labels", default="data/oxford_pet_yolo/labels/train")
    parser.add_argument("--output-dir", default="outputs/annotation_checks")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_polygons(label_path: Path, width: int, height: int):
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        coordinates = np.asarray(fields[1:], dtype=np.float32).reshape(-1, 2)
        coordinates[:, 0] *= width
        coordinates[:, 1] *= height
        polygons.append(coordinates.astype(np.int32).reshape(-1, 1, 2))
    return polygons


def main():
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    images = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {images_dir}")

    random.seed(args.seed)
    selected = random.sample(images, min(args.num_samples, len(images)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Aviso: rótulo não encontrado para {image_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Aviso: não foi possível abrir {image_path}")
            continue

        height, width = image.shape[:2]
        polygons = load_polygons(label_path, width, height)
        if polygons:
            overlay = image.copy()
            cv2.fillPoly(overlay, polygons, color=(0, 180, 0))
            image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
            cv2.polylines(image, polygons, isClosed=True, color=(0, 255, 0), thickness=2)

        output_path = output_dir / f"{image_path.stem}_annotation.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"Salvo: {output_path}")


if __name__ == "__main__":
    main()
