import copy
import json
import random
import shutil
import numpy as np
import pytest
import torch
from torch import nn
from src.checkpoint import capture_rng, load_checkpoint, restore_rng
from src.evaluate import evaluate_checkpoint
from src.train import train


def tiny_model(cfg, initialize=True):
    return nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.Dropout2d(.1), nn.Conv2d(8, 5, 1))


def test_interruption_resume_exact(raster_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.build_model", tiny_model)
    monkeypatch.setattr("src.evaluate.build_model", tiny_model)
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    train(raster_cfg, "uniform_50", 17, output_dir=a)
    train(raster_cfg, "uniform_50", 17, output_dir=b, stop_after_epochs=1)
    train(raster_cfg, "uniform_50", 17, output_dir=b, resume=True)
    ca, cb = (load_checkpoint(p / "checkpoints/last.pt") for p in (a, b))
    assert ca["history"] == cb["history"]
    for key in ca["model"]:
        torch.testing.assert_close(ca["model"][key], cb["model"][key], rtol=0, atol=0)
    assert ca["scheduler"] == cb["scheduler"]
    reloaded = evaluate_checkpoint(a / "checkpoints/best.pt")
    assert reloaded["metrics"] == json.loads((a / "best_validation.json").read_text())["metrics"]
    with pytest.raises(FileExistsError, match="Completed"):
        train(raster_cfg, "uniform_50", 17, output_dir=a, resume=True)


def test_resume_controls_and_artifacts(raster_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.build_model", tiny_model)
    output = tmp_path / "run"
    train(raster_cfg, "uniform_10", 17, output_dir=output, stop_after_epochs=1)
    changed = copy.deepcopy(raster_cfg)
    changed["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="identical"):
        train(changed, "uniform_10", 17, output_dir=output, resume=True)
    moved = tmp_path / "moved"
    shutil.copytree(output, moved)
    with pytest.raises(ValueError, match="original output directory"):
        train(raster_cfg, "uniform_10", 17, output_dir=moved, resume=True)
    (output / "checkpoints/best.pt").unlink()
    with pytest.raises(FileNotFoundError, match="preserved"):
        train(raster_cfg, "uniform_10", 17, output_dir=output, resume=True)


def test_rng_roundtrip():
    generator = torch.Generator().manual_seed(17)
    before = capture_rng(generator)
    first = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator))
    restore_rng(before, generator)
    second = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)


def test_final_test_gate(raster_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.build_model", tiny_model)
    output = train(raster_cfg, "uniform_50", 17, output_dir=tmp_path / "run")
    with pytest.raises(PermissionError):
        evaluate_checkpoint(output / "checkpoints/best.pt", split="test")


def test_real_unet_synthetic_raster_roundtrip(raster_cfg, tmp_path):
    raster_cfg["training"]["epochs"] = 1
    output = train(raster_cfg, "uniform_50", 17, output_dir=tmp_path / "unet")
    reloaded = evaluate_checkpoint(output / "checkpoints/best.pt")
    assert reloaded["metrics"] == json.loads((output / "best_validation.json").read_text())["metrics"]
    from src.visualization import examples, training_curves
    training_curves(output)
    examples(output)
    assert (output / "figures/training_curves.csv").is_file()
    assert (output / "figures/train_example.png").is_file()
    assert (output / "figures/val_example.pdf").is_file()
    evidence = json.loads((output / "figures/examples.json").read_text())
    assert not evidence["test_inference"]
