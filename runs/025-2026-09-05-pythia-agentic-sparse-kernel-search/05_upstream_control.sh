#!/usr/bin/env bash
# H100 ONLY. Run under an external timeout; source checkout is new and untouched.
# This reproduces U0's own supported FFN workload, not a Pythia speedup claim.
set -euo pipefail
run025_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run025_attempt="${1:?Pass a new attempt identifier}"
[[ "$run025_attempt" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
run025_state="$run025_dir/artifacts/u0-$run025_attempt"
mkdir -- "$run025_state"
export UV_CACHE_DIR="$run025_dir/artifacts/runtime/uv-cache"
export HF_HOME="$run025_dir/artifacts/runtime/hf-cache"
export TORCH_EXTENSIONS_DIR="$run025_state/torch_extensions"
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=2
run025_commit=661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5
run025_revision=7c2a0473ec982facd5b21af81fe39c1783f3407e
run025_source="$run025_state/upstream"
run025_model="$run025_state/SparseLM0.5B"
command -v uv
git clone --no-checkout --filter=blob:none https://github.com/SakanaAI/sparser-faster-llms.git "$run025_source"
git -C "$run025_source" -c core.autocrlf=false checkout --detach "$run025_commit"
[[ "$(git -C "$run025_source" rev-parse HEAD)" == "$run025_commit" ]]
uv venv --python python3 "$run025_state/venv"
run025_python="$run025_state/venv/bin/python"
uv pip install --python "$run025_python" --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
# Same prebuilt FlashAttention artifact used by the successful Run 022 U0.
uv pip install --python "$run025_python" 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl'
uv pip install --python "$run025_python" transformers==4.56.2 tqdm matplotlib safetensors huggingface-hub ninja pynvml setuptools pip
"$run025_python" -c 'import torch; assert torch.cuda.get_device_capability() == (9,0)'
"$run025_state/venv/bin/hf" download SakanaAI/SparseLM0.5B --revision "$run025_revision" --local-dir "$run025_model"
printf '%s\n' "$run025_revision" > "$run025_state/model-revision.txt"
(cd "$run025_model" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum) > "$run025_state/model-sha256.txt"
(cd "$run025_source" && git ls-files -z | xargs -0 sha256sum) > "$run025_state/source-sha256.txt"
"$run025_python" -m pip freeze > "$run025_state/pip-freeze.txt"
(
  cd "$run025_source"
  "$run025_python" benchmark_inference.py --model-path "$run025_model" \
    --out-csv "$run025_state/positive-control.csv" --batch-size 64 --seq-len 2048 \
    --dtype bf16 --device cuda --reps 50 --warmup-reps 5
)
git -C "$run025_source" diff --exit-code
date -u +%FT%TZ > "$run025_state/finished-utc.txt"
