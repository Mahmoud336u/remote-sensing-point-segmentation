"""Atomic, weights-only-readable checkpoints and reproducible RNG restoration."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, threads: int = 4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng(loader_generator: torch.Generator) -> dict:
    ns = np.random.get_state()
    return {"python": random.getstate(), "numpy": [ns[0], ns[1].tolist(), *ns[2:]],
            "torch": torch.get_rng_state(), "loader": loader_generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict, loader_generator: torch.Generator):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA checkpoint requires CUDA for exact resume")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, state: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return state
