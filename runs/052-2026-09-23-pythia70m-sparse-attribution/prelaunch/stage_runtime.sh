#!/usr/bin/env bash
set -euo pipefail
export PATH="/workspace/run052/049-2026-09-22-pythia70m-short-row-limits/runtime/venv/bin:$PATH"
exec bash /workspace/run052/052-2026-09-23-pythia70m-sparse-attribution/12_stage.sh "$@"
