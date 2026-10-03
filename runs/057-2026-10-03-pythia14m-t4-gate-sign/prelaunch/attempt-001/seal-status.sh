set -eu
date -u
run=/workspace/sparsity-spillover/runs/057-2026-10-03-pythia14m-t4-gate-sign
if test -f /workspace/run057-control/seal.pid; then
    ps -p "$(cat /workspace/run057-control/seal.pid)" -o pid,etime,pcpu,stat,args || true
fi
if test -f "$run/transfer/training-001.tar.gz"; then
    stat -c '%s bytes' "$run/transfer/training-001.tar.gz"
fi
if test -f "$run/transfer/training-receipt-001.json"; then
    cat "$run/transfer/training-receipt-001.json"
fi
tail -n 3 /workspace/run057-control/seal.log
