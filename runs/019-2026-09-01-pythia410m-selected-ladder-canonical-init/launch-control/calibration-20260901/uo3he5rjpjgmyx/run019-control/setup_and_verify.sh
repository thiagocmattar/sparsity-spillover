#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/sparsity-spillover
control=/workspace/run019-control
run_dir="$repo/runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init"

mkdir -p "$control"
date -u +%FT%TZ > "$control/setup-started-utc.txt"
status=0
{
    set -euo pipefail
    cd "$repo"
    test "$(git rev-parse HEAD)" = 7a0c49f54a5d0b89943228490908417952766905
    sha256sum -c <<'EOF'
da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c  data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin
bf4a2a7a96891191fdbe627e60f39be2b51f9c50024bccdc30cf10988a140693  data/tokenized/minipile-pythia-14m-full/train/metadata.json
51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451  data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin
70a2e1d487405ab1c9ba38e299a52d07066157e21ec1b8cb965706899274f451  data/tokenized/minipile-pythia-14m-full/validation/metadata.json
edba25fa7535bef6469228afe7d60db741a1a4ecd6c0c4362989e0b45a8cac51  runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/pythia410m-seed1234.safetensors
38c9f36a2e259018c183039771c5732d8aa5a5f63b4363d9e6403e6683a9ef43  runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/pythia410m-seed1234-rng.pt
b909c5c578ee264127097b7c2f226cabf982de45b9033dc2d9f8e413ae174523  runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/metadata.json
EOF
    bash "$run_dir/01_setup_remote.sh"
} > "$control/setup.log" 2>&1 || status=$?

printf '%s\n' "$status" > "$control/setup-exit-code.txt"
date -u +%FT%TZ > "$control/setup-finished-utc.txt"
exit "$status"
