"""Reusable inference utilities for img2GPS.

This module is intentionally independent from Streamlit/FastAPI so the same prediction logic
can be used by CLI scripts, notebooks, the web demo, and the API service.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, BinaryIO

import torch
from PIL import Image

from .constants import NUM_CELLS
from .model import GPSGridModel, load_checkpoint, CELL_CENTER_LATS, CELL_CENTER_LONS
from .transforms import get_eval_transform


@dataclass
class TopKCell:
    cell_id: int
    probability: float
    center_latitude: float
    center_longitude: float


@dataclass
class PredictionResult:
    latitude: float
    longitude: float
    # Top-1 grid softmax probability. This is retained as "confidence" for
    # backward compatibility with earlier API/demo code, but it should be
    # interpreted as concentration over grid cells, not calibrated GPS accuracy.
    confidence: float
    top_k_probability_mass: float
    entropy: float
    top_k_cells: list[TopKCell]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["top_k_cells"] = [asdict(cell) for cell in self.top_k_cells]
        return data


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_inference_model(checkpoint_path: str | Path, device: str | torch.device = "auto") -> GPSGridModel:
    resolved_device = get_device(str(device)) if isinstance(device, str) else device
    model = GPSGridModel(pretrained=False, freeze_backbone=True).to(resolved_device)
    load_checkpoint(model, str(checkpoint_path), map_location=resolved_device)
    model.eval()
    return model


def load_image(image: str | Path | BinaryIO | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return Image.open(image).convert("RGB")


def predict_image(
    model: GPSGridModel,
    image: str | Path | BinaryIO | Image.Image,
    top_k: int = 5,
    device: str | torch.device = "auto",
) -> PredictionResult:
    if top_k < 1 or top_k > NUM_CELLS:
        raise ValueError(f"top_k must be in [1, {NUM_CELLS}], got {top_k}")

    resolved_device = get_device(str(device)) if isinstance(device, str) else device
    pil_image = load_image(image)
    x = get_eval_transform()(pil_image).unsqueeze(0).to(resolved_device)
    model = model.to(resolved_device)

    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_cells = probs.topk(top_k, dim=1)
        normalized_top_probs = top_probs / top_probs.sum(dim=1, keepdim=True)
        lat_centers = CELL_CENTER_LATS.to(resolved_device)[top_cells]
        lon_centers = CELL_CENTER_LONS.to(resolved_device)[top_cells]
        pred_lat = (normalized_top_probs * lat_centers).sum(dim=1).item()
        pred_lon = (normalized_top_probs * lon_centers).sum(dim=1).item()
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).item())

    top_k_cells = [
        TopKCell(
            cell_id=int(cell_id),
            probability=float(prob),
            center_latitude=float(CELL_CENTER_LATS[cell_id]),
            center_longitude=float(CELL_CENTER_LONS[cell_id]),
        )
        for cell_id, prob in zip(top_cells[0].cpu().tolist(), top_probs[0].cpu().tolist())
    ]

    return PredictionResult(
        latitude=float(pred_lat),
        longitude=float(pred_lon),
        # This is the top-1 grid softmax probability, not a calibrated accuracy score.
        confidence=float(top_probs[0, 0].item()),
        top_k_probability_mass=float(top_probs[0].sum().item()),
        entropy=entropy,
        top_k_cells=top_k_cells,
    )
