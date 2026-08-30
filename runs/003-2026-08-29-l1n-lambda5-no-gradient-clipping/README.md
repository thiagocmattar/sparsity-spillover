# 003 - Pythia-14M lambda-5 L1N without gradient clipping

**State:** completed; scientifically verified with a Git-provenance limitation

## Approved question

Does disabling global gradient-norm clipping materially change the short-horizon
`lambda=5` naive-L1 outcome for GeLU and ReLU relative to the exactly matched,
clipped Run 002 conditions?

The hypothesis is deliberately nondirectional. A meaningful change in training
dynamics, terminal validation loss, activation statistics, or weight norms
supports a clipping effect; a similar outcome argues that clipping is not the
main explanation at this seed and horizon. No minimum material-effect threshold
was predeclared, so comparisons remain descriptive.

## Matched design

The two conditions are GeLU and ReLU with operational `l1_naive` pressure of
weight `5.0` applied only to all six `h` activations. They reuse Run 002's
Pythia-14M random initialization, seed-0 ordered data schedule, 451 updates,
global batch 64 as microbatch 4 x accumulation 16, 2,048-token sequences, peak
LR `4e-3`, warmup/cosine schedule, AdamW settings, BF16 autocast, FP32
parameters/state, and zero dropout. The GeLU model uses stock `h`; the ReLU
model uses topology `A1-H` with standard ReLU at `h`.

The sole optimizer intervention change is `gradient_clip_norm: null`. No
norm-based rescaling is applied. The implementation still computes the global
combined-gradient norm, rejects nonfinite gradients before AdamW, and logs that
clipping is disabled. The clipped comparators are Run 002 attempts
`009-20260829-133102-56cafec6` and
`010-20260829-134316-109e1518`.

## Validation and retained diagnostics

Each condition evaluates all 500 MiniPile validation documents after update 1
and from the reloaded final checkpoint: 338 complete blocks, 692,224 input
tokens, and the declared 1,444-token excluded tail. Every optimizer boundary
retains task/pressure gradient norms, dot product, cosine, conflict flag, and
the unscaled combined global norm. Terminal full-validation diagnostics retain
integer exact/near-zero counts at `0`, `1e-3`, and `1e-2`, plus RMS/L2 moments,
for `h`, `q_post`, `k_post`, `v`, `m`, and the post-`W_o`, pre-residual
attention output. Per-layer weight norms and hash-verified final checkpoints
are retained.

Logical-product counters and a post-hoc clipping frontier are omitted by the
approved design. Gradient interaction is collected during training because it
cannot be reconstructed from a checkpoint.

## Manuscript relationship and limitations

This ablates the operational norm-1 clipping used with the manuscript's
`l1-pressure` and `naive-objective` equations. The unclipped update is an
intentional design-material departure from the repository reference optimizer;
it does not revise the manuscript. With one seed, two activation operators,
only `lambda=5`, and 451 updates, the result cannot establish a generally
optimal clipping policy or long-horizon stability.

## Approvals

- Design confirmed by the user on 2026-08-29.
- Launch confirmed by the user on 2026-08-29 after implementation, tests,
  production-shaped smoke calibration, local fit, exact ETC, and monitoring
  behavior were reported.

## Implemented prelaunch state

The shared optimizer accepts an explicit null clip norm. In that path it
computes and validates the global accumulated-gradient norm without modifying
any parameter gradient. Run-local training, terminal verification, calibration,
and monitoring scripts implement only the approved two-condition comparison.
The verifier requires no clipping on every event, exact Run 002 initialization
and schedule hashes, complete validation, 36 activation-statistic rows, 76
weight rows, checkpoint hashes, and exact post-`W_o`/pre-residual equality.

Focused tests pass 11/11 and the full repository suite passes 72/72. Both exact
GeLU and ReLU smoke paths completed eight optimizer boundaries, with six timed
after warmup exclusion. All losses and gradients were finite, clipping stayed
disabled, checkpoint parameters survived save/hash/reload exactly, and each
diagnostic produced six pooled sites across all six layers.

The immutable calibration is
`prelaunch/calibration-20260829-153201.json`. Its ETC for 451 updates per
condition is 1,521.49 seconds median (25m21.5s) and 1,580.96 seconds p90
(26m21.0s), including setup, update-1 validation, terminal diagnostic
validation, weights, checkpoints, cache verification, and 30 seconds of
terminal headroom. The p90 remains within the approved 1,800-second local
envelope.

Peak Torch allocated/reserved memory was 6,450,265,600 / 8,489,271,296 bytes on
the 12,820,480,000-byte RTX 5070 Ti Laptop GPU. Combining peak reservation with
the 2,715 MiB prelaunch desktop allocation leaves approximately 1.38 GiB of
conservative headroom. Avoid other GPU-heavy work during execution.

The exact execution, artifact, warning, and monitoring definition is locked in
`prelaunch/launch-plan.json`.

## Execution and terminal result

The detached local cohort launched at `2026-08-29T15:37:54.9704441Z` and the
driver exited at `2026-08-29T16:03:18.344229Z`: 1,523.37 seconds (25m23.4s),
inside the approved 1,800-second envelope. Both attempts completed 451 updates,
two complete-validation passes, all retained diagnostics, and checkpoint
save/hash/reload. Together they cover 902 optimizer updates, 118,226,944
training input tokens, four complete-validation passes, and 112,560,844 bytes
of hash-verified final checkpoints.

| Activation | Final val loss, clipped | Final val loss, no clip | Delta | Max gradient norm | Updates above 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeLU | 5.615236 | 5.638119 | +0.022882 | 1.267362 | 6 |
| ReLU | 5.609286 | 5.746995 | +0.137709 | 1.470671 | 3 |

At near-zero threshold `1e-3`, the no-clipping GeLU result has 20.1632% mass at
`h` and 0.0902% mean mass across `q_post`, `k_post`, and `v`; the changes from
its clipped comparator are +1.6376 and -0.0061 percentage points. The
no-clipping ReLU result has 99.5124% at `h` and 0.0751% across those attention
sites; its changes are +0.0556 and -0.0214 percentage points. Final validation
loss is worse without clipping for both matched conditions, especially ReLU,
but this remains a one-seed, short-horizon descriptive result.

## Verification limitation

Both immutable attempt manifests contain `git_commit: null` and
`git_dirty: null`. The detached process did not persist the underlying Git
subprocess failure, so its exact cause is unknown. The original driver made
these fields fatal only after both scientific attempts had completed, producing
`driver.json` status `failed` with `Git commit missing/mixed.`

The append-only `05_verify_missing_git_provenance.py` preserves those null
fields and reuses every other original terminal check. The resulting
`artifacts/verification.json` is `verified` with evidence label
`valid_with_provenance_limitation`: both attempts match the prelaunch executed
run-code content hash, initialization and schedule hashes, configs, event
sequences, full-validation coverage, finite unclipped gradients, activation and
weight diagnostics, checkpoint contents, and transfer inventories. No attempt
artifact was rewritten. This limitation affects repository-state provenance,
not the detected scientific artifact consistency.
