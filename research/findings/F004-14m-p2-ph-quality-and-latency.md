# F004 - P2 raises 14M loss and has a small threshold-dependent latency effect

Status: confirmed

Scope: the measured recipes, checkpoints, implementation and workload

Approved: 2026-10-04, by the user's request to close the P2/Ph workstream and
record its conclusion in the workflow.

## Statement

With T2 one-sided gates at h/z, expanding OL1 pressure from h (Ph) to h/z
(P2) increases validation loss at both kappa=0 and .1. It increases empty z
tiles while reducing h sparsity. The frozen K050 inference benchmark does not
show a consistent P2 latency advantage: P2 is 0.677% slower at 0 and 1.660%
faster at .1. The small conditional gain is not a quality-matched improvement.

## Evidence and definitions

Run058 trained four Pythia-14M conditions from matched random initial bytes,
seed1234 and data order, for 712 optimizer updates. Canonical FP16 validation
loss increases by .153062/.178198 under P2 at kappa=0/.1. At .1, aligned empty
8x16 tiles increase from 20.704% to 74.828% at z and decrease from 43.696% to
34.653% at h. R_model changes from 5.1708% to 5.1970%; logical opportunity and
tile occupancy are distinct from runtime.

Run059 evaluates those retained checkpoints on one physical RTX5090, BF16,
batch1, sequence2048, uncached causal inference with all50304 logits and frozen
K050. Recurring gates, detection, packing and full logits are timed; input
staging and setup are excluded. Each threshold has 64 fixed inputs, seven
randomized paired passes and three fresh processes: 1344 direct pairs.

| kappa | Ph host geomean ms | P2 host geomean ms | Paired Ph/P2 latency ratio | P2 latency reduction |
|---:|---:|---:|---:|---:|
| 0 | 0.659690 | 0.664153 | 0.993279 | -0.677% |
| .1 | 0.625165 | 0.614790 | 1.016875 | +1.660% |

Speedup is the geometric mean of same-input/pass Ph/P2 synchronized-host
latency ratios. All three process ratios agree in direction at each threshold.
Native graph controls are effectively tied. Every implementation passes against
its own eager anchor on all 500 MiniPile validation documents: 338 complete
blocks, 692224 input tokens, 691886 predictions and a 1444-token excluded tail.
The BF16 benchmark loss penalties are .156577/.178161; they retain their own
precision identity rather than replacing Run058's canonical FP16 loss.

## Caveats and provenance

One training seed and one GPU are not population or scaling evidence. System
process repetitions are not training replications. Equal lambda with an
unweighted mean over six versus twelve pressure tensors reweights h; this is
a recipe comparison, not an isolated added-z-gradient effect. There is no
equal-loss, decoding/KV-cache or training-speed conclusion. Another kernel,
training seed or matched-quality comparison could change the trade-off.

- [Run058 observation](../../runs/058-2026-10-03-pythia14m-t2-p2-versus-ph/observations/001-p2-versus-ph.md),
  [tables](../../runs/058-2026-10-03-pythia14m-t2-p2-versus-ph/results/tables.md),
  [reducer and figure script](../../runs/058-2026-10-03-pythia14m-t2-p2-versus-ph/21_analyze.py).
- [Run059 observation](../../runs/059-2026-10-04-pythia14m-p2-ph-latency/observations/01-p2-ph-latency.md),
  [source-hashed results](../../runs/059-2026-10-04-pythia14m-p2-ph-latency/results/results.json),
  [reducer](../../runs/059-2026-10-04-pythia14m-p2-ph-latency/20_analyze.py),
  [figure script](../../runs/059-2026-10-04-pythia14m-p2-ph-latency/21_figure.py).

Both owned Pods were deleted after verified retrieval. Run059 closeout is
committed as c7995140. No additional P2/Ph experiment or manuscript edit is
requested. The next requested workstream is training with a tile threshold;
its design choices and evidence remain separate in [the backlog](../BACKLOG.md).
