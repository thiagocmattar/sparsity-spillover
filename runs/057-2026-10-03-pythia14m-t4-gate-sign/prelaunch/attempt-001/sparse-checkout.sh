set -eu
root=/workspace/sparsity-spillover
control=/workspace/run057-control
cd "$root"
git sparse-checkout set --cone src/sparsity_research runs/004-2026-08-29-pythia14m-full-pass-l1n runs/057-2026-10-03-pythia14m-t4-gate-sign > "$control/sparse-checkout.log" 2>&1
git checkout --quiet HEAD >> "$control/sparse-checkout.log" 2>&1
test -e runs/057-2026-10-03-pythia14m-t4-gate-sign/inputs/random-initialization/provenance.json
git rev-parse HEAD
