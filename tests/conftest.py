from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from src.config import load_config
from src.data_manifest import build_manifest


@pytest.fixture
def cfg():
    result = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml"))
    result["model"]["pretrained"] = False
    result["training"]["threads"] = 2
    return result


@pytest.fixture
def raster_cfg(tmp_path, cfg):
    cfg["profile"] = "synthetic_test_fixture"
    cfg["data"].update(root=str(tmp_path / "rasters"), manifest=str(tmp_path / "splits.json"),
                       patch_size=64, tile_caps={"train": 4, "val": 2, "test": 2},
                       archive_url="synthetic-fixture://local", archive_sha256="0" * 64)
    cfg["experiment"]["output_root"] = str(tmp_path / "runs")
    root = Path(cfg["data"]["root"])
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for i in range(9):
        h, w = 65 + i % 2, 79
        rng = np.random.default_rng(i)
        rgb = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
        mask = (np.indices((h, w)).sum(0) % 5).astype(np.uint8)[None]
        meta = {"driver": "GTiff", "height": h, "width": w, "dtype": "uint8",
                "crs": "EPSG:2180", "transform": from_origin(100000 + 3000 * i, 500000, 1, 1)}
        for folder, data in (("images", rgb), ("masks", mask)):
            with rasterio.open(root / folder / f"scene_{i}.tif", "w", count=len(data), **meta) as f:
                f.write(data)
    build_manifest(root, Path(cfg["data"]["manifest"]), cfg["data"])
    return cfg


@pytest.fixture(autouse=True)
def torch_threads():
    torch.set_num_threads(2)
