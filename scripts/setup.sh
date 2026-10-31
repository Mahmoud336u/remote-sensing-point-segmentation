#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export UV_CACHE_DIR=/workspace/scratch/uv-cache
export PIP_CACHE_DIR=/workspace/scratch/pip-cache
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
uv --system-certs pip sync --python .venv/bin/python --require-hashes \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match requirements-phase3-cpu.lock
.venv/bin/python -m pip check
