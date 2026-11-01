"""Execute the reconstructed assessment notebook in the project Python kernel."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

# Support both python scripts/validate_notebook.py and python -m scripts.validate_notebook.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import file_sha256, write_json


def main():
    prefix = Path("/workspace/scratch/point-segmentation-jupyter")
    os.environ["JUPYTER_PATH"] = str(prefix / "share/jupyter")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(prefix / "runtime")
    os.environ["IPYTHONDIR"] = str(prefix / "ipython")
    os.environ["MPLCONFIGDIR"] = "/workspace/scratch/matplotlib"
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(prefix),
                    "--name", "point-segmentation", "--display-name", "Point segmentation (.venv)"],
                   check=True, capture_output=True, text=True)
    path = ROOT / "notebooks/assessment_demo.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=180, kernel_name="point-segmentation",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    if not code_cells or any(c.execution_count is None for c in code_cells):
        raise RuntimeError("Not every code cell executed")
    if any(o.output_type == "error" for c in code_cells for o in c.outputs):
        raise RuntimeError("Notebook contains errors")
    figures = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs)
    if figures < 1:
        raise RuntimeError("The sampling demonstration did not render a figure")
    markers = [line.split("NOTEBOOK_SECTION_STATUS=", 1)[1]
               for cell in code_cells for out in cell.outputs
               for line in out.get("text", "").splitlines()
               if line.startswith("NOTEBOOK_SECTION_STATUS=")]
    if len(markers) != 1:
        raise RuntimeError("Notebook must report the evidence status of its ten sections")
    sections = json.loads(markers[0])
    if len(sections) != 10:
        raise RuntimeError("Notebook section evidence inventory is incomplete")
    nbformat.write(notebook, path)
    result = {"status": "passed", "code_cells_executed": len(code_cells), "rendered_figures": figures,
              "path": str(path), "sha256": file_sha256(path),
              "scope": "offline mathematical demo and available run-artifact inspection",
              "section_status": sections,
              "real_data_assessment_complete": not any(v.startswith("blocked") for v in sections.values()),
              "real_data_smoke_available": (ROOT / "results/verification/real_data_smoke.json").exists()}
    write_json(ROOT / "results/verification/notebook.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
