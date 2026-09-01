"""Geographic helper functions for grid-based image geolocation."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from .constants import GRID_N, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, LAT_STEP, LON_STEP


def validate_bounds(lat: float, lon: float) -> bool:
    """Return True when a coordinate lies inside the configured target region."""
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def get_cell(lat: float, lon: float) -> int:
    """Map a latitude/longitude pair to a grid-cell id.

    Coordinates outside the configured region are clipped to the nearest border cell.
    This makes training robust to small GPS noise near the boundary.
    """
    row = min(max(int((lat - LAT_MIN) / LAT_STEP), 0), GRID_N - 1)
    col = min(max(int((lon - LON_MIN) / LON_STEP), 0), GRID_N - 1)
    return row * GRID_N + col


def cell_center(cell_id: int) -> Tuple[float, float]:
    """Return the latitude/longitude center of a grid cell."""
    if not 0 <= cell_id < GRID_N * GRID_N:
        raise ValueError(f"cell_id must be in [0, {GRID_N * GRID_N - 1}], got {cell_id}")
    row = cell_id // GRID_N
    col = cell_id % GRID_N
    lat = LAT_MIN + (row + 0.5) * LAT_STEP
    lon = LON_MIN + (col + 0.5) * LON_STEP
    return lat, lon


def offset_from_center(lat: float, lon: float, cell_id: int) -> Tuple[float, float]:
    """Return normalized within-cell offset relative to the cell center."""
    center_lat, center_lon = cell_center(cell_id)
    return (lat - center_lat) / LAT_STEP, (lon - center_lon) / LON_STEP


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in meters between two GPS points."""
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def haversine_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Vectorized Haversine distance for tensors shaped [N, 2] in lat/lon order."""
    radius_m = 6_371_000.0
    pred_rad = torch.deg2rad(pred)
    target_rad = torch.deg2rad(target)
    d = pred_rad - target_rad
    a = torch.sin(d[:, 0] / 2).pow(2) + torch.cos(pred_rad[:, 0]) * torch.cos(target_rad[:, 0]) * torch.sin(d[:, 1] / 2).pow(2)
    return 2 * radius_m * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1 - a, min=0.0)))
