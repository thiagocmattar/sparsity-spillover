#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while test ! -e runtime/pipeline-023.exit; do sleep 10; done
test "$(cat runtime/pipeline-023.exit)" = 0
bash 05_execute.sh -c "from io_utils import read,RUN; from statistics import median; rows=read(RUN/'artifacts/development/summary-022.json'); assert len(rows)==6 and all(r['qualified'] for r in rows); assert median(r['timing']['candidate_graph']['median_host_ms'] for r in rows if r['condition']=='c21')<1.18"
bash 05_execute.sh 11_candidate.py freeze opt073 >runtime/freeze-024.log 2>&1
bash 05_execute.sh 03_execute.py --phase final --tag 024 --deadline-epoch 1789926479 >runtime/final-024.log 2>&1
bash 05_execute.sh 03_execute.py --phase decomposition --tag 024 --deadline-epoch 1789926479 >runtime/decomposition-024.log 2>&1
bash 05_execute.sh 20_summarize.py >runtime/summary-024.log 2>&1
bash 05_execute.sh 42_profile_attribution.py >runtime/attribution-024.log 2>&1
bash 05_execute.sh 43_verify_final_evidence.py >runtime/verification-024.log 2>&1
bash 05_execute.sh 08_collect.py --tag final001 >runtime/recovery-final001.log 2>&1
