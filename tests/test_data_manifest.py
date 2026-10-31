import copy
import json
from pathlib import Path
import pytest
from shapely.geometry import box, mapping
from src.data_manifest import audit_manifest, group_and_split, load_manifest


def test_inventory_and_frozen_separation(raster_cfg):
    path = Path(raster_cfg["data"]["manifest"])
    before = path.read_bytes()
    manifest = load_manifest(path, raster_cfg["data"]["root"], verify_hashes=True)
    audit = audit_manifest(manifest)
    assert audit["groups"] == audit["scene_count"] == 9
    assert min(audit["minimum_cross_partition_distance_m"].values()) > 256
    assert path.read_bytes() == before


def test_transitive_geography_and_duplicates():
    positions = [0, 200, 400, 2000, 4000, 6000, 8000, 10000, 12000, 14000]
    scenes = [{"id": str(i), "image_sha256": str(i),
               "footprint_epsg2180": mapping(box(x, 0, x + 10, 10))}
              for i, x in enumerate(positions)]
    scenes[-1]["image_sha256"] = scenes[-2]["image_sha256"]
    result = group_and_split(scenes)
    byid = {s["id"]: s for s in result}
    assert len({byid[str(i)]["group"] for i in range(3)}) == 1
    assert byid["8"]["group"] == byid["9"]["group"]
    assert result == group_and_split(scenes)


def test_hash_change_and_leakage_rejected(raster_cfg):
    path = Path(raster_cfg["data"]["manifest"])
    manifest = json.loads(path.read_text())
    a = next(s for s in manifest["scenes"] if s["split"] == "train")
    b = next(s for s in manifest["scenes"] if s["split"] == "val")
    b["group"] = a["group"]
    with pytest.raises(ValueError, match="leakage"):
        audit_manifest(manifest)
    image = Path(raster_cfg["data"]["root"]) / a["image"]
    with image.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(path, raster_cfg["data"]["root"], verify_hashes=True)
