set -eu
run=/workspace/sparsity-spillover/runs/060-2026-10-04-pythia14m-temporal-corner-tile
control=/workspace/run060-control
date -u +%FT%TZ
if [ -f "$control/seal.exit" ]; then printf 'seal.exit='; cat "$control/seal.exit"; fi
tail -n 3 "$control/seal.log" || true
if [ -f "$run/transfer/training-001.tar.gz" ]; then stat --printf='archive bytes=%s modified=%y\n' "$run/transfer/training-001.tar.gz"; fi
if [ -f "$run/transfer/training-receipt-001.json" ]; then cat "$run/transfer/training-receipt-001.json"; fi
