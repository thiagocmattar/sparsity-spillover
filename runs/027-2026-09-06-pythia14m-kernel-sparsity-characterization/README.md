# Run 027: Pythia-14M kernel speedup versus achieved sparsity

Status: full 105-process matrix running on the user-approved 14M cohort.
Authorization: 2026-09-06 request explicitly approves all execution/RunPod
needed for the per-variant scientific figure. No further approval wait is used.
Run 026 remains immutable. This new diagnostic owns its source, measurements
and figures; no manuscript prose or consolidated finding is changed.

## Question and fixed comparison

Does the frozen K019+K018 algorithm's full-model speedup increase with achieved
R_model, and what part of its latency benefit comes specifically from skipping
zero products rather than fusion? Evaluate all 30 main-study 14M endpoints in
Analysis 013 and all five historical A4+OL1@h endpoints, labeled by realized
intervention. A cross-checkpoint association is descriptive, not a causal
effect of training pressure; thresholds are treatment levels, not replicates.

No training: keep the existing seed-1234 random-pretraining checkpoint at step
712, source optimizer/schedule/data order, gate operators, thresholds, and
pressure provenance. Families: A0/GELU, A1-H/ReLU, A1-H+naive L1, A1-H+OL1,
A4, corrected A4+OL1@4, A7, A7+OL1@7, and historical A4+OL1@h. The four-site
gates are one-sided; A7's post-RoPE Q/K/V gates are symmetric. Pressure/gate
sites remain independent and imported from actual source manifests.

Pythia-14M: six layers, hidden128, FFN512, four heads x32, vocab50304.
BF16 parameters/logits, B1 T2048, complete logits, causal SDPA, no cache or
training, default cuBLAS workspace, TF32/reduced BF16 reductions disabled.
Same RTX5090 within this run; model-only transfer, no optimizer state.
Original K019 RoPE kernel is unchanged. K018 receives an explicit no-gate
compatibility path for A0/A1-H and a zero-skipping toggle; no threshold or
tile tuning. A0 preserves GELU; absent gates stay absent, not ReLU.

Modes paired on identical new inputs:

- native: checkpoint's canonical stock eager forward, no added hooks;
- fusion_dense: K019 exact RoPE/gate/layout fusion, stock dense projections;
- sparse: K019 plus K018 gate/projection/residual fusion and zero skipping;
- no_skip: same K018 fusion/rounding and inputs, but all features execute FMA,
  including zeros. This deliberately untuned control isolates skipping, not
  the fastest dense competitor. It is never labeled an optimized baseline.

Full numerical qualification: all 500 validation documents / 338 complete
2048-token blocks, 692224 inputs, 691886 prediction targets, 1444-token excluded
tail. Fixed logit bounds atol=.25, rtol=.02, relativeL2=.02; pooled loss
difference <=.001 nat/token. Failures are recorded and excluded from qualified
speedup claims, not hidden. Three fresh processes per checkpoint; 64 validation
identities with seed2504, seven randomized paired passes, warmed resident
inputs. No runtime implementation selection or per-checkpoint fallback.

Store native/fusion/sparse/no-skip latencies, all raw samples, source hashes,
integer logical numerators/denominators, qualification and loss. Canonical
R_model is inherited from the same source logical pass across all modes.
Actual BF16 h/z exact counts and occupancy are independently captured after
timing, with activation exact/near-zero counts at 0/.001/.01, RMS/L2 and weight
norms. Preserve original local checkpoint/cache identities. Gradient conflict
is training-only and remains in source records; no new clipping is added.

The manuscript question is whether measured runtime benefits accompany
model-wide logical sparsity (draft sec:measurement / sec:experiments). Use the
operational R_model = block zero products / (block products + dense LM head)
and report percent on axes; it is not measured FLOPs or a speed ceiling.
K018 exploits h/z products, not all products counted by R_model, so eligible
per-operation counts and the zero-skip-disabled comparison are essential.
Increasing qualified speedup/skip benefit with sparsity supports the mechanism;
flat or reversing behavior, slowdowns, or numerical failure limit/refute it.

## Execution and budget

Current live quote: community RTX5090 $0.69/h. Four-hour stop backstop and
$5 total task ceiling (within the prior remaining $25 authorization); expected
measurement time tens of minutes including setup/transfers, to refine after
smoke. 40GB container +40GB Pod volume; no new network volume. Data/artifacts
under /workspace, detached bounded worker, logs every checkpoint/validation
chunk. Monitor phase completions or every five minutes, sooner on exceptions,
stale events >5 minutes, nonfinite results, disk >85%, or budget risk.
Retrieve and verify all terminal artifacts before deleting compute; preserve
existing 100GB volume. Provider billing lag will be reported explicitly.

The user was asked non-blockingly for extra post-hoc measurements; the complete
diagnostic set above and retained checkpoints are the default if no reply.

## Prelaunch verification and infrastructure record

Local verification: all 242 bootstrap tests, 13 topology/optional-gate adapter
tests and two timing-reduction tests pass (257 total). The 35 checkpoint
identities, full logical coverage and source-count aggregation are verified.
The model-only input bundle is 1,825,710,912 bytes, SHA256
`044dd4115f1bd4066dba11ae4e1b8305e3e939f6376b38562186b1df95ef9f19`.
GPU primitive equivalence, smoke qualification and representative ETC remain
pending; local CPU tests do not establish CUDA equivalence.

Infrastructure-only attempts, no training or scientific input change:

- `rtx5090-001` / `w5ld1qrs54rg3x`: no container or logs after six minutes;
  deleted before any data upload or execution, independent guard cancelled.
- `rtx5090-002` / `3vds4whv0wlaqq`: same symptom after more than five minutes;
  deleted before upload/execution, guard cancelled.
- `rtx5090-003`: secure RTX 5090 quoted $0.99/hour but provider rejected the
  allocation for insufficient stock; subsequent resource list was empty.
- `rtx5090-004` / `1mzymzb5p0rcrt`: community RTX 5090 constrained to the US
  to avoid the previous France placement, $0.69/hour, independent four-hour
  stop guard. CUDA driver initialization failed before benchmark execution,
  including after one restart; seven diagnostic logs were retrieved and
  hash-verified before deletion and guard cancellation.
- `rtx5090-005` / `npmrkvv1q1l36u`: healthy Canadian community RTX5090,
  driver570.211.01, 520W power limit, $0.69/hour. Base and pinned runtimes pass
  real CUDA tensor operations. All inputs/code pass inventories. The full
  matrix runs here only; no timing from failed hosts is pooled. Four-hour
  guard ends18:23:23UTC; worker has a three-hour timeout. Setup failures count
  toward the $5 total task ceiling.

GPU verification: all32 optional-gate/skip-toggle primitive cases pass; gated
skip-enabled outputs are bitwise equal to originalK018. DenseA0 and A7+OL1@7
kappa.5 smoke tests each cover four timing inputs x two passes and eight full
validation blocks; all four modes pass every fixed numerical gate. A0 pooled
smoke loss difference is0.000138887nat/token; A7 losses are identical. These
small timings are compatibility evidence, not final study estimates. Kernel
compilation takes about40s initially and is excluded from steady-state timing.

After smoke, `prelaunch/frozen.json` seals28 model/kernel/timer/validation
source files (145052bytes); the remote copies all verify against that seal.
The 105-process matrix is launched detached via `09_matrix.sh`, with persistent
per-process events/results and `runtime/matrix-status.json`. Initial refreshed
ETC is45-75minutes, refined after the first full process. Normal read-only
monitoring interval is five minutes; first-process completion is checked sooner
to establish full-diagnostic throughput. Final full338-block qualification is
still required; no final per-variant result is claimed yet.
