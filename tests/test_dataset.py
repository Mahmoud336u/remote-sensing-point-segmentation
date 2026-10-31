import numpy as np
import pytest
import torch
from src.dataset import SegmentationDataset


def test_sparse_only_and_fixed_annotations(raster_cfg):
    ds = SegmentationDataset(raster_cfg, "train", "uniform_50", 17)
    first = ds[0]
    assert set(first) == {"image", "target", "point_mask", "valid_mask", "tile_id"}
    assert first["point_mask"].sum() == min(50, first["valid_mask"].sum())
    assert torch.all(first["target"][~first["point_mask"]] == 255)
    record = ds.annotation_record(0)
    ds.set_epoch(3)
    assert ds.annotation_record(0) == record
    assert ds[0]["point_mask"].sum() == first["point_mask"].sum()


def test_padded_edges_and_normalization(raster_cfg):
    raster_cfg["data"]["tile_caps"]["train"] = None
    raster_cfg["data"]["augment_d4"] = False
    ds = SegmentationDataset(raster_cfg, "train", "uniform_200", 17)
    item = ds[1]
    diag = ds.diagnostic_dense(1)
    assert 0 < item["valid_mask"].sum() < 64 * 64
    assert torch.all(item["target"][~item["valid_mask"]] == 255)
    mean = np.asarray(raster_cfg["data"]["mean"])[:, None, None]
    std = np.asarray(raster_cfg["data"]["std"])[:, None, None]
    np.testing.assert_allclose(item["image"].numpy() * std + mean, diag["rgb"], atol=1e-7)
    assert np.any(diag["dense_target"] == 0)


def test_dense_heldout_and_test_gate(raster_cfg):
    val = SegmentationDataset(raster_cfg, "val")
    first = val[0]
    assert torch.equal(first["point_mask"], first["valid_mask"])
    val.set_epoch(123)
    assert torch.equal(first["target"], val[0]["target"])
    with pytest.raises(PermissionError):
        SegmentationDataset(raster_cfg, "test")
    assert len(SegmentationDataset(raster_cfg, "test", allow_test=True)) == 2


def test_subsets_paired_independent_of_run(raster_cfg):
    a = SegmentationDataset(raster_cfg, "train", "uniform_10", 17)
    b = SegmentationDataset(raster_cfg, "train", "stratified_50", 43)
    assert a.tiles == b.tiles and a.subset_hash == b.subset_hash
    u50 = SegmentationDataset(raster_cfg, "train", "uniform_50", 17)
    for index in range(len(a)):
        assert set(a.annotation_record(index)["flat_indices"]) <= set(u50.annotation_record(index)["flat_indices"])


def test_dense_reference_explicit(raster_cfg):
    ds = SegmentationDataset(raster_cfg, "train", "dense_reference")
    batch = ds[0]
    assert torch.all(batch["target"][batch["valid_mask"]] < 5)
    assert torch.equal(batch["point_mask"], batch["valid_mask"])
