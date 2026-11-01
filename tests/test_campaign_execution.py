import json
from pathlib import Path
import pytest
from torch import nn

from experiments import run_experiments as runner
from src.evaluate import evaluate_checkpoint
from src.config import write_json


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
    assert result["completed_rows"][0]["experiment_id"] == "A2/B1"
    assert result["completed_rows"][0]["points_per_tile"] == 50
    assert result["completed_rows"][0]["training_seconds"] > 0
    assert 0 < result["completed_rows"][0]["occupied_cell_fraction"] <= 1
    root = Path(raster_cfg["experiment"]["output_root"])
    assert json.loads((root / "run_status.json").read_text())[0]["status"] == "completed"
    assert "completed" in (root / "run_status.csv").read_text()
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
    assert all(row["status"] == "failed" for row in runner.collect(raster_cfg)["runs"])
    assert all(row["error"] for row in json.loads((root / "run_status.json").read_text()))
    assert not (root / "results.csv").exists()


@pytest.fixture
def completed_campaign(raster_cfg, monkeypatch):
    monkeypatch.setattr("src.train.build_model", tiny)
    monkeypatch.setattr("src.evaluate.build_model", tiny)
    monkeypatch.setattr(runner, "preflight", lambda cfg: {"status": "synthetic_test_only"})
    raster_cfg["training"]["epochs"] = 1
    runner.execute(raster_cfg)
    return raster_cfg, Path(runner.plan(raster_cfg)[0]["directory"])


def test_changed_summary_checkpoint_rejected(completed_campaign):
    cfg, directory = completed_campaign
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    summary["checkpoint_sha256"] = "0" * 64
    write_json(path, summary)
    with pytest.raises(ValueError, match="different checkpoint"):
        runner.collect(cfg)


def test_corrupted_saved_validation_rejected(completed_campaign):
    cfg, directory = completed_campaign
    path = directory / "best_validation.json"
    best = json.loads(path.read_text())
    best["metrics"]["miou"] = .99
    write_json(path, best)
    with pytest.raises(ValueError, match="selection rule"):
        runner.collect(cfg)


def test_changed_final_test_controls_rejected_before_inference(completed_campaign, monkeypatch):
    cfg, directory = completed_campaign
    runner.freeze(cfg)
    cfg["training"]["lr"] *= 2
    monkeypatch.setattr(runner, "evaluate_checkpoint", lambda *a, **k: pytest.fail("Inference occurred"))
    with pytest.raises(ValueError, match="frozen configuration"):
        runner.final_test(cfg)
    assert not (directory / "final_test.json").exists()


def test_cached_final_test_validated_before_reuse(completed_campaign, monkeypatch):
    cfg, directory = completed_campaign
    runner.freeze(cfg)
    runner.final_test(cfg)
    path = directory / "final_test.json"
    result = json.loads(path.read_text())
    result["metrics"]["miou"] = .99
    write_json(path, result)
    monkeypatch.setattr(runner, "evaluate_checkpoint", lambda *a, **k: pytest.fail("Inference occurred"))
    with pytest.raises(ValueError, match="miou disagrees"):
        runner.final_test(cfg)
