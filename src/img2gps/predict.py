"""Predict GPS coordinates for local image files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from .model import GPSGridModel, load_checkpoint
from .train import get_device
from .transforms import get_eval_transform


def load_image_tensor(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return get_eval_transform()(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict GPS coordinates for images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("images", nargs="+", help="Image paths")
    args = parser.parse_args()

    device = get_device()
    model = GPSGridModel(pretrained=False, freeze_backbone=True).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    rows = []
    tensors = []
    valid_paths = []
    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)
        tensors.append(load_image_tensor(path))
        valid_paths.append(str(path))

    batch = torch.stack(tensors).to(device)
    preds = model.predict_tensor(batch, top_k=args.top_k).cpu().numpy()
    for path, (lat, lon) in zip(valid_paths, preds):
        row = {"image": path, "pred_latitude": float(lat), "pred_longitude": float(lon)}
        rows.append(row)
        print(f"{path}\tlatitude={lat:.6f}\tlongitude={lon:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "pred_latitude", "pred_longitude"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
