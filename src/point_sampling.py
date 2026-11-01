"""Fixed sparse coordinates using a documented reconstructed seed algorithm."""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

SEED_VERSION = "sha256-json-v1-reconstructed"


def stable_seed(run_seed: int, source_id: str, tile_id: str, namespace: str) -> int:
    payload = json.dumps([SEED_VERSION, int(run_seed), source_id, tile_id, namespace],
                         ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_points(valid: np.ndarray, budget: int, seed: int,
                  strategy: str = "uniform") -> np.ndarray:
    valid = np.asarray(valid)
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise ValueError("valid must be a 2D boolean mask")
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if strategy not in ("uniform", "stratified"):
        raise ValueError("Unknown sampling strategy")
    coords = np.flatnonzero(valid)
    count = min(budget, len(coords))
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = np.zeros(valid.size, dtype=bool)
    if count == 0:
        return chosen.reshape(valid.shape)
    if strategy == "uniform":
        chosen[rng.permutation(coords)[:count]] = True
    else:
        h, w = valid.shape
        bands = max(1, int(math.sqrt(budget)))
        base, extra = divmod(budget, bands)
        rows = np.linspace(0, h, bands + 1, dtype=int)
        for band in range(bands):
            columns = base + (band < extra)
            edges = np.linspace(0, w, columns + 1, dtype=int)
            for col in range(columns):
                region = valid[rows[band]:rows[band + 1], edges[col]:edges[col + 1]]
                yy, xx = np.nonzero(region)
                if len(yy):
                    i = rng.integers(len(yy))
                    chosen[(yy[i] + rows[band]) * w + xx[i] + edges[col]] = True
        remaining = count - int(chosen.sum())
        if remaining:
            candidates = coords[~chosen[coords]]
            chosen[rng.permutation(candidates)[:remaining]] = True
    return chosen.reshape(valid.shape)


def d4(array: np.ndarray, transform: int) -> np.ndarray:
    """Apply the same D4 element to the final two axes of any aligned tensor."""
    if transform not in range(8):
        raise ValueError("D4 transform must be 0..7")
    result = np.rot90(array, transform % 4, axes=(-2, -1))
    if transform >= 4:
        result = np.flip(result, axis=-1)
    return np.ascontiguousarray(result)


def spatial_coverage(valid: np.ndarray, points: np.ndarray, bins: int = 8) -> dict:
    """Class-blind geometry diagnostics; never used to select training points.

    Occupancy uses a fixed grid over the valid bounding rectangle. Nearest-point
    distances normalize each axis to [0,1]. Dense masks omit the quadratic
    nearest-neighbor calculation; their occupied-cell fraction remains defined.
    """
    valid, points = np.asarray(valid), np.asarray(points)
    if (valid.ndim != 2 or points.shape != valid.shape or valid.dtype != np.bool_
            or points.dtype != np.bool_ or np.any(points & ~valid)):
        raise ValueError("Coverage needs aligned boolean masks and valid selected points")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    vy, vx = np.nonzero(valid)
    py, px = np.nonzero(points)
    if not len(vy):
        return {"grid_bins_per_axis": bins, "valid_cells": 0, "occupied_cells": 0,
                "occupied_cell_fraction": None, "mean_nearest_neighbor_normalized": None}
    y0, x0 = vy.min(), vx.min()
    height, width = vy.max() - y0 + 1, vx.max() - x0 + 1
    valid_cells = np.unique(((vy - y0) * bins // height) * bins + (vx - x0) * bins // width)
    occupied = np.unique(((py - y0) * bins // height) * bins + (px - x0) * bins // width)
    nearest = None
    if 2 <= len(py) <= 200:
        normalized = np.column_stack(((py - y0) / max(height - 1, 1),
                                      (px - x0) / max(width - 1, 1)))
        distance = np.sqrt(((normalized[:, None] - normalized[None, :]) ** 2).sum(-1))
        np.fill_diagonal(distance, np.inf)
        nearest = float(distance.min(axis=1).mean())
    return {"grid_bins_per_axis": bins, "valid_cells": int(len(valid_cells)),
            "occupied_cells": int(len(occupied)), "occupied_cell_fraction": len(occupied) / len(valid_cells),
            "mean_nearest_neighbor_normalized": nearest}
