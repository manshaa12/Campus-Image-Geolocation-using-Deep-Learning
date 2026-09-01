"""Plot the distribution of training samples across the 10 x 10 grid.

Example:
    python scripts/plot_grid_distribution.py --output outputs/grid_distribution.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from img2gps.constants import DATASET_NAME, GRID_N
from img2gps.data import detect_coordinate_columns
from img2gps.geo import get_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="outputs/grid_distribution.png")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    lat_col, lon_col = detect_coordinate_columns(ds)
    counts = np.zeros((GRID_N, GRID_N), dtype=int)
    for item in ds:
        cell = get_cell(float(item[lat_col]), float(item[lon_col]))
        counts[cell // GRID_N, cell % GRID_N] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(counts, origin="lower")
    ax.set_title("Training image distribution across 10 x 10 grid cells")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    for row in range(GRID_N):
        for col in range(GRID_N):
            ax.text(col, row, str(counts[row, col]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Number of images")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved grid distribution plot to {output}")


if __name__ == "__main__":
    main()
