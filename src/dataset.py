"""Windowed native-resolution RGB data with sparse-only training targets."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset

from .config import CONDITIONS, canonical_hash, resolve_path
from .data_manifest import load_manifest
from .point_sampling import SEED_VERSION, d4, sample_points, stable_seed


def select_tiles(scenes: list[dict], patch_size: int, cap: int | None,
                 seed: int, split: str) -> list[dict]:
    """Class-blind fixed subset, independent of condition and training seed."""
    tiles = []
    for scene in sorted(scenes, key=lambda s: s["id"]):
        for row in range(0, scene["height"], patch_size):
            for col in range(0, scene["width"], patch_size):
                tiles.append({"scene": scene["id"], "row": row, "col": col,
                              "id": f"{scene['id']}:{row}:{col}"})
    if cap is not None and cap < len(tiles):
        rng = np.random.Generator(np.random.PCG64(stable_seed(seed, split, "", "subset")))
        indices = sorted(rng.permutation(len(tiles))[:cap])
        tiles = [tiles[i] for i in indices]
    return tiles


class SegmentationDataset(Dataset):
    def __init__(self, cfg: dict, split: str, condition: str = "uniform_50", seed: int = 17,
                 *, allow_test: bool = False):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if split == "test" and not allow_test:
            raise PermissionError("Test access requires explicit final-test authorization")
        if condition not in CONDITIONS:
            raise ValueError("Unknown condition")
        self.cfg, self.split, self.condition, self.seed = cfg, split, condition, seed
        self.epoch = 0
        self.root = resolve_path(cfg, cfg["data"]["root"])
        self.manifest_path = resolve_path(cfg, cfg["data"]["manifest"])
        self.manifest = load_manifest(self.manifest_path, self.root)
        self.scenes = {s["id"]: s for s in self.manifest["scenes"] if s["split"] == split}
        self.size = cfg["data"]["patch_size"]
        self.tiles = select_tiles(list(self.scenes.values()), self.size,
                                  cfg["data"]["tile_caps"][split],
                                  cfg["data"]["subset_seed"], split)
        if not self.tiles:
            raise ValueError(f"No {split} tiles")
        self.subset_hash = canonical_hash(self.tiles)

    def __len__(self):
        return len(self.tiles)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _read(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tile = self.tiles[index]
        scene = self.scenes[tile["scene"]]
        height = min(self.size, scene["height"] - tile["row"])
        width = min(self.size, scene["width"] - tile["col"])
        window = Window(tile["col"], tile["row"], width, height)
        rgb = np.zeros((3, self.size, self.size), dtype=np.float32)
        dense = np.full((self.size, self.size), 255, dtype=np.int64)
        with rasterio.open(self.root / scene["image"]) as im:
            rgb[:, :height, :width] = im.read(window=window).astype(np.float32) / 255
        with rasterio.open(self.root / scene["mask"]) as ma:
            dense[:height, :width] = ma.read(1, window=window)
        valid = np.zeros((self.size, self.size), dtype=bool)
        valid[:height, :width] = True
        if np.any((dense[valid] < 0) | (dense[valid] >= self.cfg["model"]["classes"])):
            raise ValueError("Invalid dense labels in source data")
        return rgb, dense, valid

    def diagnostic_dense(self, index: int) -> dict:
        """Explicit diagnostic interface; never consumed by the sparse trainer."""
        rgb, dense, valid = self._read(index)
        return {"rgb": rgb, "dense_target": dense, "valid_mask": valid,
                "tile": self.tiles[index]}

    def annotation_record(self, index: int) -> dict:
        tile = self.tiles[index]
        scene = self.scenes[tile["scene"]]
        valid = np.zeros((self.size, self.size), dtype=bool)
        valid[:min(self.size, scene["height"] - tile["row"]),
              :min(self.size, scene["width"] - tile["col"])] = True
        strategy, budget = CONDITIONS[self.condition]
        points = valid if strategy == "dense" else sample_points(
            valid, budget, stable_seed(self.seed, tile["scene"], tile["id"], strategy), strategy)
        return {"tile_id": tile["id"], "source_id": tile["scene"],
                "valid_pixels": int(valid.sum()), "annotated_pixels": int(points.sum()),
                "flat_indices": np.flatnonzero(points).tolist() if strategy != "dense" else None,
                "selection": "all_valid" if strategy == "dense" else "fixed_points"}

    def __getitem__(self, index: int) -> dict:
        rgb, dense, valid = self._read(index)
        tile = self.tiles[index]
        points = valid.copy()
        target = dense
        if self.split == "train":
            strategy, budget = CONDITIONS[self.condition]
            if strategy != "dense":
                points = sample_points(valid, budget,
                                       stable_seed(self.seed, tile["scene"], tile["id"], strategy),
                                       strategy)
                target = np.full_like(dense, 255)
                target[points] = dense[points]
        mean = np.array(self.cfg["data"]["mean"], dtype=np.float32)[:, None, None]
        std = np.array(self.cfg["data"]["std"], dtype=np.float32)[:, None, None]
        rgb = (rgb - mean) / std
        if self.split == "train" and self.cfg["data"]["augment_d4"]:
            transform = stable_seed(self.seed, tile["scene"], tile["id"],
                                    f"augmentation-epoch-{self.epoch}") % 8
            rgb, target, points, valid = [d4(a, transform) for a in (rgb, target, points, valid)]
        return {"image": torch.from_numpy(np.ascontiguousarray(rgb)),
                "target": torch.from_numpy(np.ascontiguousarray(target)),
                "point_mask": torch.from_numpy(np.ascontiguousarray(points)),
                "valid_mask": torch.from_numpy(np.ascontiguousarray(valid)),
                "tile_id": tile["id"]}

    def provenance(self) -> dict:
        return {"split": self.split, "tile_count": len(self), "subset_hash": self.subset_hash,
                "tiles": self.tiles, "seed_algorithm": SEED_VERSION}
