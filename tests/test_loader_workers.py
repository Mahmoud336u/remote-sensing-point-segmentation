import torch
from src.dataset import SegmentationDataset
from src.train import make_loader


def test_spawn_two_workers_matches_single_process(raster_cfg):
    dataset = SegmentationDataset(raster_cfg, "train")
    raster_cfg["training"]["num_workers"] = 2
    loader = make_loader(dataset, raster_cfg, torch.Generator().manual_seed(17), False)
    batch = next(iter(loader))
    for index in range(len(batch["target"])):
        torch.testing.assert_close(batch["image"][index], dataset[index]["image"], rtol=0, atol=0)
        assert torch.equal(batch["target"][index], dataset[index]["target"])
