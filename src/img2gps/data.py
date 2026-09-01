"""Dataset loading and splitting utilities for img2GPS."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, Subset

from .constants import DATASET_NAME
from .geo import get_cell, offset_from_center


class HFGPSDataset(Dataset):
    """PyTorch wrapper for Hugging Face image-to-GPS datasets.

    The wrapper returns four values: image tensor, grid cell id, raw GPS tensor, and normalized
    within-cell offset. The same class works for both training and evaluation transforms.
    """

    def __init__(self, hf_dataset, transform=None, lat_col: str = "latitude", lon_col: str = "longitude") -> None:
        self.ds = hf_dataset
        self.transform = transform
        self.lat_col = lat_col
        self.lon_col = lon_col

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        image = _load_image(item["image"])
        if self.transform is not None:
            image = self.transform(image)

        lat = float(item[self.lat_col])
        lon = float(item[self.lon_col])
        cell = get_cell(lat, lon)
        offset = torch.tensor(offset_from_center(lat, lon, cell), dtype=torch.float32)
        gps = torch.tensor([lat, lon], dtype=torch.float64)
        return image, torch.tensor(cell, dtype=torch.long), gps, offset


class CSVImageGPSDataset(Dataset):
    """Local CSV-based dataset used for optional reference-set evaluation.

    The CSV must contain an image path column plus latitude and longitude columns. Image paths
    may be absolute or relative to `image_dir` when supplied, otherwise relative to the CSV file.
    """

    def __init__(self, csv_path: str | Path, image_dir: Optional[str | Path] = None, transform=None) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.image_dir = Path(image_dir) if image_dir else self.csv_path.parent
        self.transform = transform
        self.image_col = detect_image_column(self.df.columns)
        self.lat_col, self.lon_col = detect_coordinate_columns_from_names(self.df.columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = Path(str(row[self.image_col]))
        if not image_path.is_absolute():
            image_path = self.image_dir / image_path
        image = _load_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        lat = float(row[self.lat_col])
        lon = float(row[self.lon_col])
        return image, torch.tensor([lat, lon], dtype=torch.float64), str(image_path)


def _load_image(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, (str, os.PathLike, Path)):
        return Image.open(image_obj).convert("RGB")
    # Hugging Face Image feature can sometimes return dictionaries with bytes/path.
    if isinstance(image_obj, dict):
        if image_obj.get("path"):
            return Image.open(image_obj["path"]).convert("RGB")
        if image_obj.get("bytes"):
            import io

            return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_obj)!r}")


def load_img2gps_dataset(dataset_name: str = DATASET_NAME, split: str = "train"):
    """Load the public Hugging Face dataset used by this project."""
    return load_dataset(dataset_name, split=split)


def detect_coordinate_columns(hf_dataset) -> Tuple[str, str]:
    return detect_coordinate_columns_from_names(hf_dataset.column_names)


def detect_coordinate_columns_from_names(column_names) -> Tuple[str, str]:
    columns = set(column_names)
    lat_candidates = ["latitude", "Latitude", "lat", "Lat", "LAT"]
    lon_candidates = ["longitude", "Longitude", "lon", "Lon", "LON", "lng", "Lng"]
    lat_col = next((name for name in lat_candidates if name in columns), None)
    lon_col = next((name for name in lon_candidates if name in columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not detect coordinate columns from: {list(column_names)}")
    return lat_col, lon_col


def detect_image_column(column_names) -> str:
    columns = set(column_names)
    candidates = ["image", "image_path", "filepath", "path", "file_name", "filename"]
    image_col = next((name for name in candidates if name in columns), None)
    if image_col is None:
        raise ValueError(f"Could not detect image column from: {list(column_names)}")
    return image_col


def make_train_val_indices(
    hf_dataset,
    lat_col: str,
    lon_col: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    strategy: str = "location_group",
    location_round_decimals: int = 5,
) -> tuple[list[int], list[int]]:
    """Create train/validation indices.

    `location_group` keeps images with nearly identical rounded coordinates in the same split,
    reducing leakage from repeated photos of the same physical point.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    rng = random.Random(seed)
    n = len(hf_dataset)

    if strategy == "random":
        indices = list(range(n))
        rng.shuffle(indices)
        n_val = max(1, int(n * val_fraction))
        return indices[n_val:], indices[:n_val]

    if strategy != "location_group":
        raise ValueError("strategy must be either 'random' or 'location_group'.")

    groups: dict[tuple[float, float], list[int]] = {}
    for idx in range(n):
        item = hf_dataset[idx]
        key = (round(float(item[lat_col]), location_round_decimals), round(float(item[lon_col]), location_round_decimals))
        groups.setdefault(key, []).append(idx)

    keys = list(groups.keys())
    rng.shuffle(keys)
    target_val = max(1, int(n * val_fraction))
    val_indices: list[int] = []
    train_indices: list[int] = []
    for key in keys:
        if len(val_indices) < target_val:
            val_indices.extend(groups[key])
        else:
            train_indices.extend(groups[key])
    return train_indices, val_indices


def subset_dataset(hf_dataset, indices: list[int]):
    """Use Hugging Face select when possible, otherwise fall back to PyTorch Subset."""
    if hasattr(hf_dataset, "select"):
        return hf_dataset.select(indices)
    return Subset(hf_dataset, indices)
