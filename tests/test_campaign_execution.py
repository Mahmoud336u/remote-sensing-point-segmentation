import json
from pathlib import Path
import pytest
from torch import nn

from experiments import run_experiments as runner
from src.evaluate import evaluate_checkpoint


def tiny(cfg, initialize=True):
    return nn.Sequential(nn.Conv2d(3, 5, 1))


def test_execute_freeze_final_test_and_immutable_controls(raster_cfg, monkeypatch):
    # Synthetic integration for the runner, explicitly isolated from the real-data gate.
    monkeypatch.setattr("src.train.build_model", tiny)
    monkeypatch.setattr("src.evaluate.build_model", tiny)
    monkeypatch.setattr(runner, "preflight", lambda cfg: {"status": "synthetic_test_only"})
    raster_cfg["training"]["epochs"] = 1
    result = runner.execute(raster_cfg)
    assert len(result["completed_rows"]) == 1
    assert result["completed_rows"][0]["split"] == "val"
    root = Path(raster_cfg["experiment"]["output_root"])
    assert not list(root.glob("**/final_test.json"))
    frozen = runner.freeze(raster_cfg)
    before = frozen.read_bytes()
    checkpoint = Path(runner.plan(raster_cfg)[0]["directory"]) / "checkpoints/best.pt"
    final = evaluate_checkpoint(checkpoint, "test", allow_final_test=True, protocol=frozen)
    assert final["metrics"]["valid_pixels"] > 0
    runner.final_test(raster_cfg)
    assert {r["split"] for r in runner.collect(raster_cfg)["completed_rows"]} == {"val", "test"}
    runner.freeze(raster_cfg)
    assert before == frozen.read_bytes()
    # Re-execution preserves a completed run rather than adding optimizer updates.
    run_file = checkpoint.parent.parent / "run.json"
    saved = run_file.read_bytes()
    runner.execute(raster_cfg)
    assert saved == run_file.read_bytes()
    raster_cfg["training"]["lr"] *= 2
    with pytest.raises(FileExistsError):
        runner.execute(raster_cfg)


def test_runner_records_failures_and_continues(raster_cfg, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda cfg: {"status": "synthetic_test_only"})
    raster_cfg["experiment"]["conditions"] = ["uniform_10", "uniform_50"]
    calls = []
    def failure(cfg, condition, seed, **kwargs):
        calls.append(condition)
        raise RuntimeError("deliberate synthetic failure")
    monkeypatch.setattr(runner, "train", failure)
    with pytest.raises(RuntimeError, match="2 runs failed"):
        runner.execute(raster_cfg)
    root = Path(raster_cfg["experiment"]["output_root"])
    assert calls == ["uniform_10", "uniform_50"]
    assert len(json.loads((root / "failures.json").read_text())) == 2
    assert not (root / "results.csv").exists()
