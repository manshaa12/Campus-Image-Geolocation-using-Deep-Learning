"""Evaluate a trained img2GPS checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import DATASET_NAME
from .data import CSVImageGPSDataset, HFGPSDataset, detect_coordinate_columns, load_img2gps_dataset
from .metrics import coordinate_distances, summarize_distances
from .model import GPSGridModel, load_checkpoint
from .transforms import get_eval_transform
from .train import get_device


def _collate_reference(batch):
    images, gps, paths = zip(*batch)
    return torch.stack(images), torch.stack(gps), list(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate img2GPS checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--csv", default=None, help="Optional local CSV with image paths and GPS labels")
    parser.add_argument("--image-dir", default=None, help="Base directory for images referenced by --csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--predictions-out", default=None, help="Optional CSV path for per-image predictions")
    args = parser.parse_args()

    device = get_device()
    model = GPSGridModel(pretrained=False, freeze_backbone=True).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    if args.csv:
        dataset = CSVImageGPSDataset(args.csv, image_dir=args.image_dir, transform=get_eval_transform())
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate_reference)
        rows = []
        preds, targets = [], []
        with torch.no_grad():
            for images, gps_raw, paths in loader:
                batch_preds = model.predict_tensor(images.to(device), top_k=args.top_k).cpu().numpy()
                for path, pred, target in zip(paths, batch_preds, gps_raw.numpy()):
                    preds.append(pred.tolist())
                    targets.append(target.tolist())
                    rows.append({"image": path, "pred_latitude": pred[0], "pred_longitude": pred[1], "latitude": target[0], "longitude": target[1]})
    else:
        raw_ds = load_img2gps_dataset(args.dataset, args.split)
        lat_col, lon_col = detect_coordinate_columns(raw_ds)
        dataset = HFGPSDataset(raw_ds, transform=get_eval_transform(), lat_col=lat_col, lon_col=lon_col)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        rows = []
        preds, targets = [], []
        with torch.no_grad():
            for batch_idx, (images, _, gps_raw, _) in enumerate(loader):
                batch_preds = model.predict_tensor(images.to(device), top_k=args.top_k).cpu().numpy()
                for i, (pred, target) in enumerate(zip(batch_preds, gps_raw.numpy())):
                    sample_id = batch_idx * args.batch_size + i
                    preds.append(pred.tolist())
                    targets.append(target.tolist())
                    rows.append({"sample_id": sample_id, "pred_latitude": pred[0], "pred_longitude": pred[1], "latitude": target[0], "longitude": target[1]})

    distances = coordinate_distances(preds, targets)
    metrics = summarize_distances(distances)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Samples: {metrics.samples}")
    print(f"Average Haversine distance: {metrics.mean_m:.2f} m")
    print(f"Median Haversine distance: {metrics.median_m:.2f} m")
    print(f"P75/P90 distance: {metrics.p75_m:.2f} m / {metrics.p90_m:.2f} m")
    print(f"Max distance: {metrics.max_m:.2f} m")

    if args.predictions_out:
        out_path = Path(args.predictions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for row, distance in zip(rows, distances):
            row["haversine_m"] = distance
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
