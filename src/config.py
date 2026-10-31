"""Validated experiment configuration and canonical provenance helpers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

CONDITIONS = {
    "uniform_10": ("uniform", 10),
    "uniform_50": ("uniform", 50),
    "uniform_200": ("uniform", 200),
    "stratified_50": ("stratified", 50),
    "dense_reference": ("dense", None),
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Replace a JSON file atomically; forbid nonstandard NaN/Infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def resolve_path(cfg: dict, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_repo_root"]) / p


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    cfg["_repo_root"] = str(path.parent.parent)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for key in ("data", "model", "training", "experiment", "profile"):
        if key not in cfg:
            raise ValueError(f"Missing configuration section: {key}")
    d, m, t, e = (cfg[k] for k in ("data", "model", "training", "experiment"))
    positive_ints = [
        d["patch_size"], m["in_channels"], m["classes"], t["epochs"],
        t["batch_size"], t["threads"],
    ]
    if any(type(v) is not int or v <= 0 for v in positive_ints):
        raise ValueError("Sizes and counts must be positive integers")
    if m["classes"] != 5 or m["in_channels"] != 3:
        raise ValueError("LandCover.ai profiles require RGB and five classes")
    if d["ignore_index"] != 255:
        raise ValueError("LandCover.ai ignore/padding value must be 255")
    if type(t["num_workers"]) is not int or t["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if t["scheduler"] not in ("cosine", "constant"):
        raise ValueError("Unknown scheduler")
    if t["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Unknown device")
    if not math.isfinite(t["lr"]) or t["lr"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 <= t["min_lr"] <= t["lr"]:
        raise ValueError("Invalid cosine minimum learning rate")
    patience = t.get("early_stopping_patience")
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError("early_stopping_patience must be positive or null")
    if len(d["mean"]) != 3 or len(d["std"]) != 3 or any(v <= 0 for v in d["std"]):
        raise ValueError("Invalid RGB normalization")
    if not e["conditions"] or any(c not in CONDITIONS for c in e["conditions"]):
        raise ValueError("Unknown or empty experiment conditions")
    if len(set(e["conditions"])) != len(e["conditions"]):
        raise ValueError("Duplicate conditions")
    if not e["seeds"] or any(type(s) is not int or s < 0 for s in e["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if len(set(e["seeds"])) != len(e["seeds"]):
        raise ValueError("Duplicate seeds")
    for cap in d["tile_caps"].values():
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("Tile caps must be positive or null")
    fractions = d["split_fractions"]
    if len(fractions) != 3 or any(v <= 0 for v in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Invalid split fractions")
    if d["split_distance_m"] < 0:
        raise ValueError("Split distance cannot be negative")
