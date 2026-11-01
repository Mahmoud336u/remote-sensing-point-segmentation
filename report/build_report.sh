#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dot -Tpdf report/figures/workflow.dot -o report/figures/workflow.pdf
dot -Tpng -Gdpi=150 report/figures/workflow.dot -o report/figures/workflow.png
pandoc report/technical_report.md --resource-path=report --standalone \
  --pdf-engine=xelatex --include-in-header=report/header.tex \
  -o report/technical_report.pdf
