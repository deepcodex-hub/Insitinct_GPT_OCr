#!/usr/bin/env bash
# run_acceptance.sh
# Runs full pipeline on benchmark/100_images and outputs acceptance_report.json + debug artifacts.

echo "[INFO] Running Automated Acceptance & Evaluation Suite on Prototype..."
export PYTHONPATH=.

# Execute Python evaluation script
.venv/Scripts/python.exe scripts/generate_acceptance.py --input benchmark/100 --out results/

echo "[INFO] Acceptance Suite execution complete."
echo "Check results/acceptance_report.json and results/failure_analysis.md"
