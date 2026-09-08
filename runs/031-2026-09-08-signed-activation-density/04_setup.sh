#!/usr/bin/env bash
# On the approved Pod, from /workspace/run031, with Python 3.12.
set -euo pipefail
run_dir=runs/031-2026-09-08-signed-activation-density
python -m venv /workspace/run031/venv
python_bin=/workspace/run031/venv/bin/python
"$python_bin" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$python_bin" -m pip install -r "$run_dir/requirements.txt"
"$python_bin" "$run_dir/03_verify.py" --inputs
mkdir -p "$run_dir/artifacts"
"$python_bin" -m pip freeze > "$run_dir/artifacts/environment.txt"
nvidia-smi > "$run_dir/artifacts/gpu.txt"
PYTHONPATH=src "$python_bin" -m pytest -p no:cacheprovider "$run_dir/test_density.py" -q
