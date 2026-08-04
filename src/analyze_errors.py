"""Salva as predições de menor confiança para análise qualitativa de erros.

Uso:
    python -m src.analyze_errors --weights outputs/runs/train/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Localiza casos de baixa confiança.")
    parser.add_argument("--weights", default="outputs/runs/train/weights/best.pt")
    parser.add_argument("--source", default="data/oxford_pet_yolo/images/val")
    parser.add_argument("--output-dir", default="outputs/error_cases")
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Quantidade de imagens enviada por chamada de inferência.")
    parser.add_argument("--conf", type=float, default=0.1)
    return parser.parse_args()


def lowest_confidence(result):
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0
    return float(result.boxes.conf.min().item())


def main():
    args = parse_args()
    source_dir = Path(args.source)
    images = sorted(
        path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {source_dir}")

    model = YOLO(args.weights)
    results = []
    for start in range(0, len(images), args.batch_size):
        batch = [str(path) for path in images[start:start + args.batch_size]]
        results.extend(model.predict(batch, conf=args.conf, save=False, verbose=False))

    worst_results = sorted(results, key=lowest_confidence)[:args.num_cases]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for position, result in enumerate(worst_results, start=1):
        score = lowest_confidence(result)
        stem = Path(result.path).stem
        output_path = output_dir / f"{position:02d}_{stem}_conf_{score:.3f}.jpg"
        result.save(filename=str(output_path))
        print(f"Salvo (confiança mínima={score:.3f}): {output_path}")


if __name__ == "__main__":
    main()
