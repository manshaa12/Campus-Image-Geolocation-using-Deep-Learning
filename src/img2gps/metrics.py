"""Evaluation metrics for img2GPS."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

from .geo import haversine_m


@dataclass
class DistanceMetrics:
    samples: int
    mean_m: float
    median_m: float
    p75_m: float
    p90_m: float
    max_m: float

    def to_dict(self):
        return asdict(self)


def summarize_distances(distances: Sequence[float]) -> DistanceMetrics:
    if len(distances) == 0:
        raise ValueError("Cannot summarize an empty distance list.")
    arr = np.asarray(distances, dtype=float)
    return DistanceMetrics(
        samples=int(arr.size),
        mean_m=float(arr.mean()),
        median_m=float(np.median(arr)),
        p75_m=float(np.percentile(arr, 75)),
        p90_m=float(np.percentile(arr, 90)),
        max_m=float(arr.max()),
    )


def coordinate_distances(preds: Iterable[Sequence[float]], targets: Iterable[Sequence[float]]) -> list[float]:
    return [haversine_m(float(t[0]), float(t[1]), float(p[0]), float(p[1])) for p, t in zip(preds, targets)]
