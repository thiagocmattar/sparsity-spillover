#!/usr/bin/env bash
set -euo pipefail

condition=${1:?condition id required}
expected_commit=${2:?expected source commit required}
repo=/workspace/sparsity-spillover
control=/workspace/run021-control
run_rel=runs/021-2026-09-03-pythia410m-a0-learning-rate-screen-resolved-lr

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test "$(cat "$control/setup-exit-code.txt")" = 0
cd "$repo"
sha256sum -c <<'EOF'
da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c  data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin
bf4a2a7a96891191fdbe627e60f39be2b51f9c50024bccdc30cf10988a140693  data/tokenized/minipile-pythia-14m-full/train/metadata.json
51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451  data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin
70a2e1d487405ab1c9ba38e299a52d07066157e21ec1b8cb965706899274f451  data/tokenized/minipile-pythia-14m-full/validation/metadata.json
edba25fa7535bef6469228afe7d60db741a1a4ecd6c0c4362989e0b45a8cac51  runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/pythia410m-seed1234.safetensors
38c9f36a2e259018c183039771c5732d8aa5a5f63b4363d9e6403e6683a9ef43  runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/pythia410m-seed1234-rng.pt
b909c5c578ee264127097b7c2f226cabf982de45b9033dc2d9f8e413ae174523  runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/metadata.json
EOF
test -x /workspace/run021-venv/bin/python
/workspace/run021-venv/bin/python -c \
  'import torch, transformers; assert torch.__version__.split("+", 1)[0] == "2.11.0"; assert torch.version.cuda == "12.8"; assert transformers.__version__ == "5.12.1"'
/workspace/run021-venv/bin/python -c \
  'import torch; assert torch.cuda.is_available(); name=torch.cuda.get_device_name(0); allowed=("RTX PRO 6000", "A100", "H100", "H200"); assert any(token in name for token in allowed), name; assert torch.cuda.get_device_properties(0).total_memory >= 75 * 1024**3; assert torch.backends.cuda.is_flash_attention_available()'
/workspace/run021-venv/bin/python "$repo/$run_rel/08_verify_initialization.py" \
  --output "$control/initialization-verification-$condition.json"
/workspace/run021-venv/bin/python "$repo/$run_rel/09_verify_worker_entrypoint.py" \
  --worker "$condition" > "$control/worker-entrypoint-$condition.json"
grep -q '"base_run_worker_dispatches_to_resolution_wrapper": true' \
  "$control/worker-entrypoint-$condition.json"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader \
  > "$control/device-$condition.txt"
