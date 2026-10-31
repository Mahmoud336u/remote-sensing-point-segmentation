"""Inventory original scenes and freeze leakage-controlled geographic partitions."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from shapely.geometry import Polygon, mapping, shape

from .config import canonical_hash, file_sha256, write_json


def inventory(root: str | Path) -> list[dict]:
    root = Path(root)
    images = sorted(root.glob("**/images/*.tif"))
    if not images:
        raise FileNotFoundError(f"No original GeoTIFFs found under {root}/**/images")
    scenes = []
    for image in images:
        mask = image.parent.parent / "masks" / image.name
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for {image.name}")
        with rasterio.open(image) as im, rasterio.open(mask) as ma:
            if im.count != 3 or im.dtypes != ("uint8",) * 3:
                raise ValueError(f"{image.name}: expected uint8 RGB")
            if (ma.count, ma.width, ma.height) != (1, im.width, im.height):
                raise ValueError(f"{image.name}: image/mask dimensions differ")
            if not np.issubdtype(np.dtype(ma.dtypes[0]), np.integer):
                raise ValueError(f"{mask.name}: labels must be integers")
            if im.crs is None:
                raise ValueError(f"{image.name}: image CRS is missing")
            if im.crs != ma.crs or im.transform != ma.transform:
                raise ValueError(f"{image.name}: image/mask georeferencing differs")
            labels = set()
            for _, window in ma.block_windows(1):
                labels.update(int(v) for v in np.unique(ma.read(1, window=window)))
            if not labels <= set(range(5)):
                raise ValueError(f"{mask.name}: unexpected labels {labels}")
            corners = [(0, 0), (im.width, 0), (im.width, im.height), (0, im.height)]
            perimeter = []
            for a, b in zip(corners, corners[1:] + corners[:1]):
                for t in np.linspace(0, 1, 17, endpoint=False):
                    perimeter.append(im.transform @ (a[0] + t * (b[0] - a[0]),
                                                      a[1] + t * (b[1] - a[1])))
            xs, ys = transform(im.crs, "EPSG:2180", *zip(*perimeter))
            footprint = Polygon(zip(xs, ys))
            if not footprint.is_valid or footprint.area <= 0:
                raise ValueError(f"{image.name}: invalid footprint")
            scenes.append({
                "id": image.stem, "image": str(image.relative_to(root)),
                "mask": str(mask.relative_to(root)), "width": im.width, "height": im.height,
                "source_crs_wkt": im.crs.to_wkt(), "transform": list(im.transform)[:6],
                "footprint_epsg2180": mapping(footprint),
                "pixel_size": [math.hypot(im.transform.a, im.transform.d),
                               math.hypot(im.transform.b, im.transform.e)],
                "image_sha256": file_sha256(image), "mask_sha256": file_sha256(mask),
                "labels_present": sorted(labels),
            })
    if len({s["id"] for s in scenes}) != len(scenes):
        raise ValueError("Duplicate scene IDs")
    return scenes


def group_and_split(scenes: list[dict], seed: int = 2026, distance_m: float = 256,
                    fractions: tuple = (.70, .15, .15)) -> list[dict]:
    """Split by connected geographic components, including duplicate-image edges."""
    scenes = [dict(s) for s in sorted(scenes, key=lambda s: s["id"])]
    parent = list(range(len(scenes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    footprints = [shape(s["footprint_epsg2180"]) for s in scenes]
    for i, j in itertools.combinations(range(len(scenes)), 2):
        if (footprints[i].distance(footprints[j]) <= distance_m or
                scenes[i]["image_sha256"] == scenes[j]["image_sha256"]):
            parent[find(j)] = find(i)
    groups = {}
    for i, scene in enumerate(scenes):
        groups.setdefault(find(i), []).append(scene)
    ordered = sorted(groups.values(), key=lambda g: tuple(s["id"] for s in g))
    if len(ordered) < 3:
        raise ValueError("At least three independent groups are required")
    n_train = int(len(ordered) * fractions[0])
    n_val = round(len(ordered) * fractions[1])
    if min(n_train, n_val, len(ordered) - n_train - n_val) < 1:
        raise ValueError("Split fractions produce an empty partition")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(ordered))
    for position, idx in enumerate(order):
        split = "train" if position < n_train else "val" if position < n_train + n_val else "test"
        group = canonical_hash(sorted(s["id"] for s in ordered[idx]))[:16]
        for scene in ordered[idx]:
            scene.update(group=group, split=split)
    return scenes


def audit_manifest(manifest: dict) -> dict:
    scenes = manifest["scenes"]
    if len({s["id"] for s in scenes}) != len(scenes):
        raise ValueError("Duplicate scene IDs")
    splits = {p: [s for s in scenes if s["split"] == p] for p in ("train", "val", "test")}
    if sum(map(len, splits.values())) != len(scenes) or not all(splits.values()):
        raise ValueError("Invalid or empty partitions")
    minimum = {}
    for a, b in itertools.combinations(splits, 2):
        for field in ("id", "group", "image_sha256"):
            if {s[field] for s in splits[a]} & {s[field] for s in splits[b]}:
                raise ValueError(f"Cross-partition {field} leakage: {a}/{b}")
        distance = min(shape(x["footprint_epsg2180"]).distance(shape(y["footprint_epsg2180"]))
                       for x in splits[a] for y in splits[b])
        minimum[f"{a}-{b}"] = distance
        if distance <= manifest["split_policy"]["distance_m"]:
            raise ValueError(f"Cross-partition geographic leakage: {a}/{b}")
    return {
        "scene_count": len(scenes),
        "groups": len({s["group"] for s in scenes}),
        "scenes_by_split": {p: len(ss) for p, ss in splits.items()},
        "groups_by_split": {p: len({s["group"] for s in ss}) for p, ss in splits.items()},
        "minimum_cross_partition_distance_m": minimum,
        "disjoint_scene_ids_groups_and_image_hashes": True,
    }


def build_manifest(root: Path, output: Path, data_cfg: dict) -> dict:
    if output.exists():
        raise FileExistsError(f"Frozen manifest already exists: {output}")
    scenes = group_and_split(inventory(root), data_cfg["split_seed"],
                             data_cfg["split_distance_m"], tuple(data_cfg["split_fractions"]))
    result = {
        "schema_version": 1,
        "provenance": "reconstructed; original manifest was not recovered",
        "source": {"url": data_cfg["archive_url"], "sha256": data_cfg["archive_sha256"]},
        "split_policy": {"seed": data_cfg["split_seed"],
                         "distance_m": data_cfg["split_distance_m"],
                         "fractions": data_cfg["split_fractions"],
                         "algorithm": "sorted-components-pcg64-floor-train-round-val-v1"},
        "scenes": scenes,
    }
    result["audit"] = audit_manifest(result)
    write_json(output, result)
    return result


def load_manifest(path: str | Path, root: str | Path, verify_hashes: bool = False) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unknown manifest format; preserve original and migrate explicitly")
    audit_manifest(manifest)
    root = Path(root).resolve()
    for scene in manifest["scenes"]:
        for kind in ("image", "mask"):
            path = (root / scene[kind]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or unsafe scene path: {scene[kind]}")
            if verify_hashes and file_sha256(path) != scene[f"{kind}_sha256"]:
                raise ValueError(f"Scene checksum mismatch: {scene['id']} {kind}")
    return manifest
