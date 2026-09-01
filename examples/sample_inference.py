"""Minimal example for local image inference.

Usage:
    python examples/sample_inference.py checkpoints/model.pt path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

# Allows running this example directly from the repository root without installing.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from img2gps.model import GPSGridModel
from img2gps.transforms import get_eval_transform
from img2gps.train import get_device


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python examples/sample_inference.py checkpoints/model.pt path/to/image.jpg")

    checkpoint = sys.argv[1]
    image_paths = sys.argv[2:]
    device = get_device()
    transform = get_eval_transform()

    model = GPSGridModel(pretrained=False, freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    images = [transform(Image.open(path).convert("RGB")) for path in image_paths]
    batch = torch.stack(images).to(device)

    with torch.no_grad():
        predictions = model.predict_tensor(batch).cpu().tolist()

    for path, (lat, lon) in zip(image_paths, predictions):
        print(f"{path}: latitude={lat:.6f}, longitude={lon:.6f}")


if __name__ == "__main__":
    main()
