# Autonomous continuation of Run 025

Status 2026-09-06: **Blackwell trajectory, 36-checkpoint frozen matrix,
fresh-process replication, component probes, fixed-policy H100 transfer, and
direct fixed-`R_model` confirmation are complete, retrieved, and verified. No
Pod or endpoint remains. The study retains an explicit partial label because
the strongest compiled-dense comparator and a CUDA-qualified sparse QK/PV path
were not completed.** See the run-level `README.md` and Analysis 015 for the
reconciled result.

2026-09-05: the user explicitly authorized the full closed-loop search, all
GPU launches and transfers, within the **same $40 total RunPod budget**.
This supersedes the pilot-only authorization boundary, not the mathematical
contract or evidence standards. Previous goal turn: **progress** (CUDA evidence,
failure diagnosis, verified retrieval and teardown), not a successful search.

The pilot remains immutable at commit `0080a30`. New candidate and evaluator
revisions live here; existing weights, config and pilot sources do not change.
The existing interactive agent/team is used; no new paid model API is provisioned.
Unexposed model request IDs/token counters remain unknown, not reconstructed.

## Objective and fixed design

Correct full-model speedups on retained Pythia 14M/70M/410M; test whether
speedup associates with canonical R_model; compare optimized vs minimally
adapted Sakana at the same checkpoint/R_model. FFN and actual QK/PV attention
were in scope; the retained K002 attention attempt passed CPU mathematics but
did not qualify on CUDA, so all final QK/PV paths remain dense. Keep B1,T2048,
BF16, full LM logits, all original gates and
weights, 18 endpoint development checkpoints and 18 untouched interior tests.
All complete validation uses 338 blocks/500 documents and the declared tail.
The current manuscript calls R_model model-wide sparsity; no TeX edits here.

The desired empirical direction is not guaranteed. Report failed conditions,
negative/null associations and dense fallbacks. Do not weaken dense references,
relax numerical gates, alter masks/thresholds, or select favorable final points.
An optimization must improve the same function, not create additional sparsity.

## Work list

- [x] Recheck current worktree and resources; zero Pods/endpoints initially.
- [x] Delegate independent numerical, one-kernel, and strong-dense probes.
- [x] Prepare hashed small 14M input bundle locally before rental.
- [x] Use bounded development leases to diagnose real activation rounding and
  test materialized P0, site-selective P0, and fused-compaction K001.
- [ ] Complete the strongest compiled/graph dense comparison.
- [x] Qualify a correct P0 revision; keep original failed source/evidence.
- [x] Seal fixed score/gates and evaluator identities; native eager remained the dense reference.
- [x] Iterate logged K001--K016 candidates on the endpoint development sets.
- [ ] Implement a qualified sparse QK/PV attention path.
- [x] Freeze architecture-specific winners and test all 36 checkpoints without feeding final scores back.
- [x] Repeat paired timing, component ablations and matched H100 transfer.
- [x] Plot R_model/speedup and fixed-R_model optimization gains with uncertainty.
- [x] Verify all Blackwell/H100/fixed-`R_model` evidence locally, reconcile the account-window budget, and terminate GPU compute.

## Spend and execution rules

Debit $1.40 provisionally for the completed pilot; initial remaining allowance
$38.60. Live account balance at 16:07 UTC was $41.2538 (not the experiment
budget). Keep at least $12 of the remaining allowance protected for final
evaluation, hardware transfer and teardown until larger-size ETC is measured.
First development lease maximum **4 RTX-5090 hours at $0.69/h = $2.76 GPU**,
plus disks/storage. This is a safety cap, not a forecast. Fall back only after
recomputing duration inside the same allocation. All GPU idle, transfers and
failed compiles count. Start the 14M work first; transfer larger checkpoints
only once the local-to-Pod link and short evaluation are verified.

Use explicit file allowlists and SHA256, compressed legacy SCP where useful,
one GPU worker at a time, detached timeout-bounded jobs, and the independent
local stop guard. Initial probes should take seconds/minutes after setup:
monitor their projected completion window, not an automatic ten-minute sleep.
Workers emit durable terminal state; no restart solely because observation
timed out. Retain final results/source snapshots before Pod deletion.

The numeric probe and initial K001 are pre-score diagnostics until the dense
baseline/evaluator is sealed. Record their cost, code and failures regardless;
do not retroactively call them held-out final evidence or a frozen trajectory.
