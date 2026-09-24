set -euo pipefail
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i /root/.ssh/run054-transfer-key -p 54754 root@103.196.86.60 'tar -xf - -C /workspace/sparsity-spillover' < /workspace/run054-control/cache-to-workers.tar
ssh -o BatchMode=yes -i /root/.ssh/run054-transfer-key -p 54754 root@103.196.86.60 'for n in $(seq 1 60); do test -f /workspace/run054-control/environment-ready && break; sleep 5; done; set -e; test -f /workspace/run054-control/environment-ready; /workspace/run054-venv/bin/python /workspace/sparsity-spillover/runs/054-2026-09-24-pythia31m-t2-ph/12_check_inputs.py; nohup setsid /workspace/run054-venv/bin/python /workspace/sparsity-spillover/runs/054-2026-09-24-pythia31m-t2-ph/15_parallel_train.py --workers hz-h-ol1-kappa-0p1 --gpus 0 --deadline 2026-09-24T19:01:00Z --tag parallel-2-002 > /workspace/run054-control/pipeline.log 2>&1 < /dev/null & echo $! > /workspace/run054-control/pipeline.pid'
rm -f /root/.ssh/run054-transfer-key
echo REPLACEMENT_STARTED
