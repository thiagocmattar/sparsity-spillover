#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/sparsity-spillover}"
PHASE="${2:-sentinel}"
STATE_ROOT="${3:-/workspace/run023-state}"
RUN_DIR="$REPO_ROOT/runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels"
OFFICIAL_DIR="$STATE_ROOT/sparser-faster-llms-official"
DERIVED_DIR="$STATE_ROOT/sparser-faster-llms-derived"
UPSTREAM_VENV="$STATE_ROOT/venv-upstream"
PYTHIA_VENV="$STATE_ROOT/venv-pythia"
UPSTREAM_COMMIT="661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5"
PATCH_PATH="$RUN_DIR/sakana-pythia70.patch"

if [[ "$PHASE" != "sentinel" && "$PHASE" != "remainder" ]]; then
  echo "phase must be sentinel or remainder" >&2
  exit 2
fi

mkdir -p "$STATE_ROOT"
PYTHON_BIN="$(command -v python3.12 || command -v python3)"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

if [[ ! -d "$OFFICIAL_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/SakanaAI/sparser-faster-llms.git "$OFFICIAL_DIR"
fi
git -C "$OFFICIAL_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$OFFICIAL_DIR" checkout --detach "$UPSTREAM_COMMIT"
if [[ ! -d "$DERIVED_DIR/.git" ]]; then
  git clone --no-hardlinks "$OFFICIAL_DIR" "$DERIVED_DIR"
fi
git -C "$DERIVED_DIR" checkout --detach "$UPSTREAM_COMMIT"

"$PYTHON_BIN" -m venv "$PYTHIA_VENV"
"$PYTHIA_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$PYTHIA_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
"$PYTHIA_VENV/bin/python" -m pip install transformers==5.12.1 numpy==2.5.0 PyYAML safetensors ninja
"$PYTHIA_VENV/bin/python" -m pip install --no-deps -e "$REPO_ROOT"

"$PYTHIA_VENV/bin/python" "$RUN_DIR/01_static_preflight.py" --phase "$PHASE" --upstream-dir "$OFFICIAL_DIR" --output "$STATE_ROOT/static-preflight-clean.json"
if git -C "$DERIVED_DIR" apply --check --whitespace=nowarn "$PATCH_PATH"; then
  git -C "$DERIVED_DIR" apply --whitespace=nowarn "$PATCH_PATH"
elif git -C "$DERIVED_DIR" apply --reverse --check --whitespace=nowarn "$PATCH_PATH"; then
  echo "Exact Run-023 patch already applied."
else
  echo "Derived tree is not the pinned clean or exact patched state." >&2
  exit 1
fi
"$PYTHIA_VENV/bin/python" "$RUN_DIR/01_static_preflight.py" --phase "$PHASE" --upstream-dir "$DERIVED_DIR" --output "$STATE_ROOT/static-preflight-patched.json"
git -C "$DERIVED_DIR" diff --binary > "$STATE_ROOT/applied-sakana-pythia70.patch"
sha256sum "$STATE_ROOT/applied-sakana-pythia70.patch" > "$STATE_ROOT/applied-sakana-pythia70.patch.sha256"

"$PYTHON_BIN" -m venv "$UPSTREAM_VENV"
"$UPSTREAM_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$UPSTREAM_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
"$UPSTREAM_VENV/bin/python" -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl
"$UPSTREAM_VENV/bin/python" -m pip install transformers==4.56.2 tqdm matplotlib safetensors huggingface-hub ninja pynvml

MODEL_DIR="$STATE_ROOT/SparseLM0.5B-7c2a047"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  "$UPSTREAM_VENV/bin/hf" download SakanaAI/SparseLM0.5B --revision 7c2a0473ec982facd5b21af81fe39c1783f3407e --local-dir "$MODEL_DIR"
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
printf '%s\n' "$OFFICIAL_DIR" > "$STATE_ROOT/upstream-official-dir.txt"
printf '%s\n' "$DERIVED_DIR" > "$STATE_ROOT/upstream-derived-dir.txt"
printf '%s\n' "$MODEL_DIR" > "$STATE_ROOT/upstream-model-dir.txt"
printf '%s\n' "$PHASE" > "$STATE_ROOT/phase.txt"
printf 'ready\n' > "$STATE_ROOT/setup-status.txt"

echo "Run 023 $PHASE official control, derived source, and immutable inputs are ready."
