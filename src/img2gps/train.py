"""Train the img2GPS grid-localization model.

Examples:
    python -m img2gps.train --config configs/default.yaml
    python -m img2gps.train --epochs 3 --batch-size 16 --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import Config, load_config
from .constants import DATASET_NAME
from .data import (
    CSVImageGPSDataset,
    HFGPSDataset,
    detect_coordinate_columns,
    load_img2gps_dataset,
    make_train_val_indices,
    subset_dataset,
)
from .metrics import coordinate_distances, summarize_distances
from .model import GPSGridModel
from .transforms import get_eval_transform, get_train_transform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_reference(batch):
    images, gps, paths = zip(*batch)
    return torch.stack(images), torch.stack(gps), list(paths)


@torch.no_grad()
def evaluate_grid_dataset(model: GPSGridModel, loader: DataLoader, device: torch.device, top_k: int = 5) -> dict:
    model.eval()
    preds, targets = [], []
    for images, _, gps_raw, _ in loader:
        pred = model.predict_tensor(images.to(device), top_k=top_k).cpu().numpy()
        preds.extend(pred.tolist())
        targets.extend(gps_raw.numpy().tolist())
    distances = coordinate_distances(preds, targets)
    return summarize_distances(distances).to_dict()


@torch.no_grad()
def evaluate_reference_dataset(model: GPSGridModel, loader: DataLoader, device: torch.device, top_k: int = 5) -> dict:
    model.eval()
    preds, targets = [], []
    for images, gps_raw, _ in loader:
        pred = model.predict_tensor(images.to(device), top_k=top_k).cpu().numpy()
        preds.extend(pred.tolist())
        targets.extend(gps_raw.numpy().tolist())
    distances = coordinate_distances(preds, targets)
    return summarize_distances(distances).to_dict()


def build_scheduler(optimizer: optim.Optimizer, epochs: int, warmup_epochs: int):
    if epochs <= 1 and warmup_epochs <= 0:
        return None

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, model: GPSGridModel, optimizer: optim.Optimizer, epoch: int, cfg: Config, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
            "metrics": metrics,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train img2GPS model")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset name")
    parser.add_argument("--split", default=None, help="Dataset split")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--split-strategy", choices=["location_group", "random"], default=None)
    parser.add_argument("--reference-csv", default=None)
    parser.add_argument("--reference-image-dir", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict = {"data": {}, "train": {}}
    if args.dataset is not None:
        overrides["data"]["dataset"] = args.dataset
    if args.split is not None:
        overrides["data"]["split"] = args.split
    if args.val_fraction is not None:
        overrides["data"]["val_fraction"] = args.val_fraction
    if args.split_strategy is not None:
        overrides["data"]["split_strategy"] = args.split_strategy
    if args.reference_csv is not None:
        overrides["data"]["reference_csv"] = args.reference_csv
    if args.reference_image_dir is not None:
        overrides["data"]["reference_image_dir"] = args.reference_image_dir
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["train"]["num_workers"] = args.num_workers
    if args.seed is not None:
        overrides["train"]["seed"] = args.seed
    if args.output_dir is not None:
        overrides["train"]["output_dir"] = args.output_dir
    if args.run_name is not None:
        overrides["train"]["run_name"] = args.run_name
    if args.no_pretrained:
        overrides["train"]["pretrained"] = False
    if args.unfreeze_backbone:
        overrides["train"]["freeze_backbone"] = False
    overrides = {k: v for k, v in overrides.items() if v}
    cfg = load_config(args.config, overrides)

    seed_everything(cfg.train.seed)
    device = get_device()
    print(f"Using device: {device}")

    raw_ds = load_img2gps_dataset(cfg.data.dataset, cfg.data.split)
    lat_col, lon_col = detect_coordinate_columns(raw_ds)
    train_indices, val_indices = make_train_val_indices(
        raw_ds,
        lat_col=lat_col,
        lon_col=lon_col,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.train.seed,
        strategy=cfg.data.split_strategy,
        location_round_decimals=cfg.data.location_round_decimals,
    )
    train_raw = subset_dataset(raw_ds, train_indices)
    val_raw = subset_dataset(raw_ds, val_indices)
    print(f"Loaded {len(raw_ds)} samples from {cfg.data.dataset}; train={len(train_raw)}, val={len(val_raw)}")
    print(f"Coordinate columns: {lat_col}, {lon_col}; split strategy: {cfg.data.split_strategy}")

    train_dataset = HFGPSDataset(train_raw, transform=get_train_transform(), lat_col=lat_col, lon_col=lon_col)
    val_dataset = HFGPSDataset(val_raw, transform=get_eval_transform(), lat_col=lat_col, lon_col=lon_col)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    ref_loader: Optional[DataLoader] = None
    if cfg.data.reference_csv:
        ref_dataset = CSVImageGPSDataset(cfg.data.reference_csv, cfg.data.reference_image_dir, transform=get_eval_transform())
        ref_loader = DataLoader(
            ref_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            collate_fn=_collate_reference,
            pin_memory=device.type == "cuda",
        )
        print(f"Reference set enabled: {len(ref_dataset)} samples from {cfg.data.reference_csv}")

    model = GPSGridModel(pretrained=cfg.train.pretrained, freeze_backbone=cfg.train.freeze_backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    offset_criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": cfg.train.lr_backbone},
            {"params": model.cell_head.parameters(), "lr": cfg.train.lr_head},
            {"params": model.offset_head.parameters(), "lr": cfg.train.lr_head},
        ],
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg.train.epochs, cfg.train.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp and device.type == "cuda")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{cfg.train.run_name}_best.pt"
    last_path = output_dir / f"{cfg.train.run_name}_last.pt"
    log_path = output_dir / f"{cfg.train.run_name}_history.jsonl"
    best_metric = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, cells, _, offsets in train_loader:
            images = images.to(device, non_blocking=True)
            cells = cells.to(device, non_blocking=True)
            offsets = offsets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp and device.type == "cuda"):
                cell_logits, pred_offsets = model(images)
                loss = cls_criterion(cell_logits, cells) + cfg.train.offset_loss_weight * offset_criterion(pred_offsets, offsets)
            if not torch.isfinite(loss):
                print("Skipping non-finite loss batch")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_grid_dataset(model, val_loader, device, top_k=cfg.train.top_k)
        monitor_metrics = val_metrics
        monitor_name = "val_mean_m"
        if ref_loader is not None:
            ref_metrics = evaluate_reference_dataset(model, ref_loader, device, top_k=cfg.train.top_k)
            monitor_metrics = ref_metrics
            monitor_name = "reference_mean_m"
        else:
            ref_metrics = None

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "validation": val_metrics,
            "reference": ref_metrics,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        save_checkpoint(last_path, model, optimizer, epoch, cfg, record)
        marker = ""
        current_metric = float(monitor_metrics["mean_m"])
        if current_metric < best_metric:
            best_metric = current_metric
            save_checkpoint(best_path, model, optimizer, epoch, cfg, record)
            marker = " saved-best"

        ref_msg = f" | ref_mean={ref_metrics['mean_m']:.2f}m" if ref_metrics else ""
        print(
            f"Epoch {epoch:02d}/{cfg.train.epochs} | loss={train_loss:.4f} | "
            f"val_mean={val_metrics['mean_m']:.2f}m | val_median={val_metrics['median_m']:.2f}m"
            f"{ref_msg} | monitor={monitor_name}{marker}",
            flush=True,
        )

    print(f"Best monitored distance: {best_metric:.2f}m")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {log_path}")


if __name__ == "__main__":
    main()
