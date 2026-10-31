import copy
from pathlib import Path
import pytest
from src.config import load_config, validate_config


@pytest.mark.parametrize("name", ["default", "cpu_pilot", "cpu_paired", "smoke"])
def test_profiles_load(name):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / f"{name}.yaml")
    assert cfg["data"]["ignore_index"] == 255
    assert cfg["model"]["classes"] == 5


@pytest.mark.parametrize("section,key,value", [
    ("data", "ignore_index", 0), ("training", "epochs", 0),
    ("training", "scheduler", "bad"), ("training", "num_workers", -1),
    ("experiment", "conditions", ["bad"]), ("experiment", "seeds", [17, 17]),
    ("model", "in_channels", 4), ("data", "split_fractions", [.5, .5, .5]),
])
def test_invalid_configs(cfg, section, key, value):
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)
