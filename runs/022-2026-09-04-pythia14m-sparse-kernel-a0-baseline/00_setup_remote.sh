#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/sparsity-spillover}"
RUN_DIR="$REPO_ROOT/runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline"
STATE_ROOT="${2:-/workspace/run022-state}"
UPSTREAM_DIR="$STATE_ROOT/sparser-faster-llms"
UPSTREAM_VENV="$STATE_ROOT/venv-upstream"
PYTHIA_VENV="$STATE_ROOT/venv-pythia"
UPSTREAM_COMMIT="661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5"

mkdir -p "$STATE_ROOT"
PYTHON_BIN="$(command -v python3.12 || command -v python3)"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/SakanaAI/sparser-faster-llms.git "$UPSTREAM_DIR"
fi
git -C "$UPSTREAM_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_COMMIT"

"$PYTHON_BIN" -m venv "$PYTHIA_VENV"
"$PYTHIA_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$PYTHIA_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
"$PYTHIA_VENV/bin/python" -m pip install transformers==5.12.1 numpy==2.5.0 PyYAML safetensors ninja
"$PYTHIA_VENV/bin/python" -m pip install --no-deps -e "$REPO_ROOT"

"$PYTHON_BIN" -m venv "$UPSTREAM_VENV"
"$UPSTREAM_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$UPSTREAM_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
"$UPSTREAM_VENV/bin/python" -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl
"$UPSTREAM_VENV/bin/python" -m pip install transformers==4.56.2 tqdm matplotlib safetensors huggingface-hub ninja pynvml

"$PYTHIA_VENV/bin/python" "$RUN_DIR/01_static_preflight.py" \
  --upstream-dir "$UPSTREAM_DIR" --output "$STATE_ROOT/static-preflight.json"

MODEL_DIR="$STATE_ROOT/SparseLM0.5B-7c2a047"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  "$UPSTREAM_VENV/bin/hf" download SakanaAI/SparseLM0.5B \
    --revision 7c2a0473ec982facd5b21af81fe39c1783f3407e \
    --local-dir "$MODEL_DIR"
fi
RESOLVED_REVISION="$("$UPSTREAM_VENV/bin/python" -c "from huggingface_hub import HfApi; print(HfApi().model_info('SakanaAI/SparseLM0.5B', revision='7c2a0473ec982facd5b21af81fe39c1783f3407e').sha)")"
if [[ "$RESOLVED_REVISION" != "7c2a0473ec982facd5b21af81fe39c1783f3407e" ]]; then
  echo "Unexpected SparseLM0.5B revision: $RESOLVED_REVISION" >&2
  exit 1
fi
printf '%s\n' "$RESOLVED_REVISION" > "$STATE_ROOT/upstream-model-revision.txt"
(cd "$MODEL_DIR" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum) > "$STATE_ROOT/upstream-model-sha256.txt"

"$PYTHIA_VENV/bin/python" -m pip freeze > "$STATE_ROOT/pythia-pip-freeze.txt"
"$UPSTREAM_VENV/bin/python" -m pip freeze > "$STATE_ROOT/upstream-pip-freeze.txt"
nvidia-smi -q > "$STATE_ROOT/nvidia-smi-q.txt"
printf '%s\n' "$UPSTREAM_DIR" > "$STATE_ROOT/upstream-dir.txt"
printf '%s\n' "$MODEL_DIR" > "$STATE_ROOT/upstream-model-dir.txt"
printf 'ready\n' > "$STATE_ROOT/setup-status.txt"

echo "Run 022 environments and immutable inputs are ready."
