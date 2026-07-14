"""Baixa o Oxford-IIIT Pet (via torchvision) e converte os trimaps de
segmentação em rótulos de polígono no formato YOLO-seg (ultralytics),
usando uma única classe: 'pet'.

Uso:
    python -m src.prepare_dataset --output-dir data/oxford_pet_yolo
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

MIN_CONTOUR_AREA = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Converte Oxford-IIIT Pet para formato YOLO-seg.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="data/oxford_pet_yolo")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def trimap_to_contours(trimap: np.ndarray):
    """Trimap oficial: 1 = pet, 2 = fundo, 3 = borda. Tratamos pet+borda
    como primeiro plano para fechar o contorno do animal."""
    mask = (trimap != 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


def contour_to_yolo_line(contour: np.ndarray, width: int, height: int, class_id: int = 0) -> str:
    points = contour.reshape(-1, 2).astype(np.float64)
    points[:, 0] /= width
    points[:, 1] /= height
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {coords}"


def export_split(base_dataset, indices, images_dir: Path, labels_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0

    for idx in tqdm(indices):
        image, trimap = base_dataset[int(idx)]
        width, height = image.size
        contours = trimap_to_contours(np.array(trimap))

        if not contours:
            skipped += 1
            continue

        largest = max(contours, key=cv2.contourArea)
        stem = f"img_{idx:05d}"
        image.convert("RGB").save(images_dir / f"{stem}.jpg", quality=90)
        (labels_dir / f"{stem}.txt").write_text(
            contour_to_yolo_line(largest, width, height) + "\n"
        )

    if skipped:
        print(f"Aviso: {skipped} imagens sem contorno válido foram ignoradas.")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    trainval = OxfordIIITPet(root=args.data_root, split="trainval",
                              target_types="segmentation", download=True)
    test = OxfordIIITPet(root=args.data_root, split="test",
                          target_types="segmentation", download=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(trainval))
    val_size = int(len(trainval) * args.val_fraction)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    print(f"Treino: {len(train_idx)} imagens | Validação: {len(val_idx)} imagens | Teste: {len(test)} imagens")

    export_split(trainval, train_idx, out_dir / "images/train", out_dir / "labels/train")
    export_split(trainval, val_idx, out_dir / "images/val", out_dir / "labels/val")
    export_split(test, range(len(test)), out_dir / "images/test", out_dir / "labels/test")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pet"},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"Dataset YOLO-seg pronto em: {out_dir}")


if __name__ == "__main__":
    main()
