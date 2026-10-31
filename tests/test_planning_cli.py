import json
from pathlib import Path
import subprocess
import sys
import pytest
from experiments.run_experiments import plan, preflight, freeze, collect
from experiments.analyze_results import summarize
from src.config import load_config


def test_plan_priority_no_side_effects(tmp_path, cfg):
    cfg["experiment"].update(output_root=str(tmp_path / "campaign"),
                             seeds=[17, 29, 43],
                             conditions=["uniform_10", "uniform_50", "uniform_200",
                                         "stratified_50", "dense_reference"])
    jobs = plan(cfg)
    assert len(jobs) == 15
    assert [j["condition"] for j in jobs[:4]] == cfg["experiment"]["conditions"][:4]
    assert all(j["condition"] == "dense_reference" for j in jobs[-3:])
    assert len({j["directory"] for j in jobs}) == len(jobs)
    assert not (tmp_path / "campaign").exists()
    assert all(r["status"] == "not_started" for r in collect(cfg)["runs"])
    with pytest.raises(RuntimeError, match="complete"):
        freeze(cfg)


def test_cli_plans_without_dataset():
    result = subprocess.run([sys.executable, "-m", "experiments.run_experiments",
                             "--config", "configs/cpu_paired.yaml"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = json.loads(result.stdout)
    assert result["planning_only"] and len(result["runs"]) == 12


def test_preflight_rejects_missing_evidence(cfg, tmp_path):
    cfg["_repo_root"] = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        preflight(cfg)


def test_single_seed_no_fabricated_std():
    row = {"condition": "uniform_50", "split": "val", "seed": 17, "annotation_fraction": .001,
           "miou": .2, "mean_dice": .3, "pixel_accuracy": .7,
           **{f"iou_class_{i}": None if i == 4 else .2 for i in range(5)}}
    one = summarize([row], "val")[0]
    assert one["miou"] == .2 and one["miou_std"] is None and one["n"] == 1
    two = summarize([row, {**row, "seed": 29, "miou": .4}], "val")[0]
    assert two["miou_std"] == pytest.approx(2 ** .5 / 10)
    assert two["iou_class_4"] is None and two["iou_class_4_n"] == 0
