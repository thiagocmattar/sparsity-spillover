# Run 027: Pythia-14M kernel speedup versus achieved sparsity

Status: complete; all 105 processes and returned evidence verified locally,
both publication PDFs rendered and visually inspected, all new Pods terminated.
Authorization: 2026-09-06 request explicitly approves all execution/RunPod
needed for the per-variant scientific figure. No further approval wait is used.
Run 026 remains immutable. This new diagnostic owns its source, measurements
and figures; no manuscript prose or consolidated finding is changed.

## Results and deliverables

All 35 checkpoints completed three fresh processes, 338-block numerical
validation per process, and 64 inputs x seven paired timing passes. The fixed
fused sparse kernel qualifies on **6/35 checkpoints** (five main-study plus one
historical endpoint); all six are faster than their own stock eager baseline.
Five high-threshold endpoints have bitwise-identical validation logits. The
remaining qualified A1-H+OL1 endpoint passes the declared tolerance bounds.

| Qualified variant | Dose | Canonical R_model (%) | Native / sparse (95% CI) | No-skip / sparse |
|---|---:|---:|---:|---:|
| A1-H+OL1 | lambda 0.1 | 3.339 | 1.490 [1.486, 1.494] | 1.531 |
| A4 | kappa 0.5 | 10.216 | 1.561 [1.557, 1.564] | 1.243 |
| A4+OL1@4 | kappa 0.5 | 12.713 | 1.561 [1.556, 1.567] | 1.228 |
| A7 | kappa 0.5 | 15.387 | 1.803 [1.799, 1.809] | 1.218 |
| A7+OL1@7 | kappa 0.5 | 27.483 | 1.808 [1.802, 1.812] | 1.236 |
| A4+OL1@h (historical) | kappa 0.5 | 10.227 | 1.562 [1.559, 1.564] | 1.243 |

The other 29 checkpoints fail the elementwise logit gate on 1-68 distinct
validation blocks, despite all loss deltas being below 0.000092 nat/token and
all relative-L2 errors below 0.00304. Their measured ratios remain visible as
open crossed points, not qualified speedup claims. QKV-fusion-only qualifies
on all 35 checkpoints. No tolerance was relaxed and no failed checkpoint was
silently given a different implementation.

For A7+OL1@7 kappa 0.5, the 1.8076x overall speedup includes a 1.2365x
skip-toggle benefit (95% CI 1.2222-1.2601). The toggle saves 0.4862 ms of the
1.6685 ms paired-mean total saving, about 29.1%; the rest is the net
non-skipping implementation difference, not a pure sparsity effect. The
no-skip kernel is an untuned mechanistic control, not an optimized dense rival.
Canonical R_model increases from 15.39% to 27.48% between the two qualified A7
kappa 0.5 endpoints, while speedup stays near 1.8x and native BF16 h/z eligible
sparsity stays near 99.9%. R_model alone is not a proportional runtime predictor.

- [All 35 variant timings and qualification](results/per-variant.md)
- [Complete numerical table](results/summary.csv) and [all four modes](results/all-variants.csv)
- [Main figure](figures/01-speedup-vs-rmodel.pdf), with [observation/caption](observations/01-speedup-vs-rmodel.md)
- [Fusion and eligible-sparsity controls](figures/02-fusion-and-eligible-sparsity.pdf), with [observation/caption](observations/02-fusion-and-eligible-sparsity.md)

The figures retain canonical source FP16-autocast R_model, explicitly distinct
from BF16 timing and native BF16 eligible-operand diagnostics. Language-model
quality is not matched across different trained variants; each speed ratio
compares implementations of the same checkpoint.

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
That canonical pass uses FP16 autocast; it is not relabeled as a newly
measured BF16 R_model. The figures and observations disclose this distinction.
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
At that prelaunch stage, GPU checks and representative ETC were pending; they
are now complete below. Local CPU tests alone did not establish CUDA equivalence.

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
  driver 570.211.01, 520 W power limit, $0.69/hour. Base and pinned runtimes pass
  real CUDA tensor operations. All inputs/code pass inventories. The full
  matrix runs here only; no timing from failed hosts is pooled. Four-hour
  guard ends 18:23:23 UTC; worker has a three-hour timeout. Setup failures count
  toward the $5 total task ceiling.

GPU verification: all 32 optional-gate/skip-toggle primitive cases pass; gated
skip-enabled outputs are bitwise equal to original K018. Dense A0 and A7+OL1@7
kappa 0.5 smoke tests each cover four timing inputs x two passes and eight full
validation blocks; all four modes pass every fixed numerical gate. A0 pooled
smoke loss difference is 0.000138887 nat/token; A7 losses are identical. These
small timings are compatibility evidence, not final study estimates. Kernel
compilation takes about 40 s initially and is excluded from steady-state timing.

After smoke, `prelaunch/frozen.json` seals 28 model/kernel/timer/validation
source files (145,052 bytes); the remote copies all verify against that seal.
The 105-process matrix is launched detached via `09_matrix.sh`, with persistent
per-process events/results and `runtime/matrix-status.json`. Initial refreshed
ETC is 45-75 minutes, refined after the first full process. Normal read-only
monitoring interval is five minutes; first-process completion is checked sooner
to establish full-diagnostic throughput. Actual matrix elapsed time was
2620.635 seconds (43.68 minutes); all 105 processes completed without an
execution failure. Numerical rejections are outcomes, not execution failures.

## Final verification and cloud closeout

The final reducer verifies every retained checkpoint/data/provenance hash,
the frozen 28-file implementation, 35 x 3 process identities, complete numerical
coverage, 188160 full-logit timing samples, every paired cell, and exact-zero
counts against per-layer row occupancy. All 258 local tests pass. PDF QA used
Poppler renders and visual inspection of both final single-page PDFs.

The returned archive is 10869769 bytes, SHA256
`bde3fb7367f55397be27e05a6cc2f888f170db2ebf2ea5581fbbcbe0e0f9cafa`.
All 689 files / 79462432 bytes pass the local transfer inventory. Source model
checkpoints remain in their original local run folders; none were deleted.
After these checks, Pod `npmrkvv1q1l36u` was terminated and guard PID47948
cancelled. Final account audit: zero Pods, zero endpoints, only the unchanged
100GB `9luykg5yc3` network volume. See `launch-control/*/closeout.json`.

Conservative estimated new task expense is **$0.928 / $5**: $0.913 GPU plus
$0.0147 Pod storage, including failed starts and restart time. This uses
observed lease durations/live GPU rates and the documented running-storage
rate with a 720-hour month ([RunPod pricing](https://docs.runpod.io/pods/pricing)).
The pre-existing volume remains a separate $7/month commitment. Provider
posting is incomplete: the snapshot contains only $0.1924 from the three failed
hosts and omits the successful Pod, so it is **not the final bill**. The full
query, GPU/disk split, assumptions and estimate are in `launch-control/billing.json`.
