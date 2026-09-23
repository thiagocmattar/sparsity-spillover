#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run052/incoming
python3 14_verify_archive.py --archive input-001.tar.gz --inventory transfer-inventory.json --receipt bundle.json --report remote-verification.json
tar -xzf input-001.tar.gz -C /workspace/run052
bash /workspace/run052/049-2026-09-22-pythia70m-short-row-limits/04_setup.sh
python3 /workspace/run052/052-2026-09-23-pythia70m-sparse-attribution/02_prepare.py verify
touch /workspace/run052/deployment-ready
