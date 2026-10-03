set -eu
pgrep -af 'git (clone|checkout)' || true
df -h /workspace /
du -sh /workspace/sparsity-spillover/.git /workspace/sparsity-spillover/runs 2>/dev/null || true
test ! -f /workspace/sparsity-spillover/.git/HEAD || cat /workspace/sparsity-spillover/.git/HEAD
tail -n 3 /workspace/run057-control/guard.log
