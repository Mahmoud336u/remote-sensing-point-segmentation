"""Record non-secret hardware and installed package versions."""
from __future__ import annotations
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import torch
from src.config import write_json


def inspect() -> dict:
    packages = ["torch", "torchvision", "segmentation-models-pytorch", "numpy", "Pillow",
                "PyYAML", "rasterio", "shapely", "matplotlib", "pytest", "nbformat", "nbclient"]
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "executable": sys.executable,
              "packages": {p: importlib.metadata.version(p) for p in packages},
              "cuda_available": torch.cuda.is_available(), "cuda_devices": torch.cuda.device_count()}
    for name in ["cpu.max", "memory.max"]:
        path = Path("/sys/fs/cgroup") / name
        result[name] = path.read_text().strip() if path.exists() else None
    return result


if __name__ == "__main__":
    result = inspect()
    write_json("docs/environment_current.json", result)
    print(json.dumps(result, indent=2))
