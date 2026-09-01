"""Preprocessing helpers for local CSV/image inputs.

The main training pipeline loads the public Hugging Face dataset directly. This file is kept for
users who want to run inference or evaluation on a local CSV file containing image paths and,
optionally, GPS labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image

from .transforms import get_eval_transform


def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    return next((name for name in candidates if name in df.columns), None)


def prepare_data(csv_path: str) -> Tuple[torch.Tensor, List[List[float]]]:
    """Load images and labels from a CSV file.

    Expected image columns include one of: image_path, filepath, image, path, file_name.
    Expected coordinate columns include latitude/longitude, Latitude/Longitude, or lat/lon.
    If coordinates are not present, dummy labels [0.0, 0.0] are returned.
    """
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file)

    img_col = _find_first_column(df, ["image_path", "filepath", "image", "path", "file_name"])
    if img_col is None:
        raise KeyError(f"No image path column found in {df.columns.tolist()}")

    lat_col = _find_first_column(df, ["latitude", "Latitude", "lat"])
    lon_col = _find_first_column(df, ["longitude", "Longitude", "lon"])

    transform = get_eval_transform()
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = Path(str(row[img_col]))
        if not image_path.is_absolute():
            image_path = csv_file.parent / image_path
        image = Image.open(image_path).convert("RGB")
        images.append(transform(image))

        if lat_col is not None and lon_col is not None:
            labels.append([float(row[lat_col]), float(row[lon_col])])
        else:
            labels.append([0.0, 0.0])

    return torch.stack(images), labels
