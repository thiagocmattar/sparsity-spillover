#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-h100nvl-001
run025_incoming="$run025_base/incoming"
run025_repo="$run025_base/sparsity-spillover"
run025_run="$run025_repo/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_bootstrap="$run025_base/bootstrap-control"
mkdir -p "$run025_bootstrap"
trap 'code=$?; printf "%s\n" "$code" > "$run025_bootstrap/exit-code.txt"; date -u +%FT%TZ > "$run025_bootstrap/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_bootstrap/started-utc.txt"

cd "$run025_incoming"
sha256sum -c SHA256SUMS > "$run025_bootstrap/incoming-sha256.txt"
mkdir "$run025_repo"
tar -xzf payload.tar.gz -C "$run025_repo"
tar -xf run025-70m-development.tar -C "$run025_repo"
tar -xf run025-410m-development.tar -C "$run025_repo"
# The current committed code is overlaid last; the 14M bundle contains the
# older calibration implementation alongside its immutable input payload.
tar -xzf run025-h100-code.tar.gz -C "$run025_repo"
date -u +%FT%TZ > "$run025_bootstrap/extracted-utc.txt"

run025_state="$run025_run/artifacts/runtime-h100nvl-001"
mkdir -p "$run025_state"
export UV_CACHE_DIR="$run025_state/uv-cache"
export PIP_CACHE_DIR="$run025_state/pip-cache"
export HF_HOME="$run025_state/hf-cache"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_run/artifacts/torch_extensions_h100nvl_001"
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
command -v nvcc
command -v g++
command -v timeout
command -v setsid
if ! command -v uv >/dev/null; then
  python3 -m pip install --break-system-packages uv==0.8.15
fi
if [[ ! -x "$run025_state/venv-pythia/bin/python" ]]; then
  uv venv --python python3 "$run025_state/venv-pythia"
fi
run025_python="$run025_state/venv-pythia/bin/python"
uv pip install --python "$run025_python" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python "$run025_python" transformers==5.12.1 numpy==2.5.0 PyYAML==6.0.2 safetensors==0.8.0 ninja==1.11.1.4 setuptools==80.9.0 pip==25.2
export PATH="$run025_state/venv-pythia/bin:/usr/local/cuda/bin:$PATH"

"$run025_python" -m py_compile \
  "$run025_run/autoresearch/probe_frozen_final.py" \
  "$run025_run/autoresearch/probe_frozen_component.py" \
  "$run025_run/autoresearch/probe_frozen_ffn.py" \
  "$run025_run/autoresearch/probe_frozen_attention_projection.py"
"$run025_python" - <<'PY'
import torch
from pathlib import Path
import sys

run = Path('/workspace/run025-h100nvl-001/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search')
sys.path.insert(0, str(run / 'autoresearch'))
from probe_frozen_component import component_sites

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] == 9
assert component_sites('ffn', 'k013', {'topology_id': 'A4-Z', 'active_sites': ['a', 'm', 'h', 'z']}) == frozenset(('m', 'h'))
assert component_sites('attention_projection', 'k016', {'topology_id': 'A7-Z-POST', 'active_sites': ['h', 'z']}) == frozenset(('z',))
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY
"$run025_python" -m pip freeze > "$run025_bootstrap/pip-freeze.txt"
nvidia-smi -q > "$run025_bootstrap/nvidia-smi-q.txt"
nvcc --version > "$run025_bootstrap/nvcc-version.txt"
date -u +%FT%TZ > "$run025_bootstrap/runtime-ready-utc.txt"

timeout --signal=TERM --kill-after=30s 9000s bash \
  "$run025_run/autoresearch/launch-control/h100nvl-001/phase16-h100-frozen-transfer.sh"

run025_phase="$run025_run/artifacts/phase16-h100-frozen-transfer-h100nvl-001"
mkdir "$run025_phase/runtime-provenance"
cp "$run025_bootstrap/pip-freeze.txt" "$run025_phase/runtime-provenance/"
cp "$run025_bootstrap/nvidia-smi-q.txt" "$run025_phase/runtime-provenance/"
cp "$run025_bootstrap/nvcc-version.txt" "$run025_phase/runtime-provenance/"
cp "$run025_bootstrap/incoming-sha256.txt" "$run025_phase/runtime-provenance/"
date -u +%FT%TZ > "$run025_phase/runtime-provenance/packaged-utc.txt"

"$run025_python" - <<'PY'
import hashlib
import json
from pathlib import Path

base = Path('/workspace/run025-h100nvl-001/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search')
paths = []
for directory in sorted((base / 'artifacts').glob('*h100nvl-001')):
    if not directory.is_dir() or directory.name.startswith(('runtime-', 'torch_extensions')):
        continue
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            paths.append({
                'path': path.relative_to(base).as_posix(),
                'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
out = base / 'artifacts/phase16-h100-frozen-transfer-h100nvl-001/evidence-inventory.json'
out.write_text(json.dumps({'files': paths}, indent=2) + '\n', encoding='utf-8')
PY

mapfile -d '' run025_artifacts < <(
  find "$run025_run/artifacts" -mindepth 1 -maxdepth 1 -type d \
    -name '*h100nvl-001' ! -name 'runtime-*' ! -name 'torch_extensions*' \
    -printf 'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/artifacts/%f\0' | sort -z
)
if [[ "${#run025_artifacts[@]}" -ne 93 ]]; then
  printf 'Expected 93 phase/evaluator artifact directories, found %s\n' "${#run025_artifacts[@]}" >&2
  exit 1
fi
run025_archive="$run025_base/run025-h100nvl-001-evidence.tar"
tar -cf "$run025_archive" -C "$run025_repo" "${run025_artifacts[@]}"
sha256sum "$run025_archive" > "$run025_archive.sha256"
date -u +%FT%TZ > "$run025_bootstrap/evidence-ready-utc.txt"
