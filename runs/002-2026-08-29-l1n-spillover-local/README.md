# 002 - Local Pythia-14M L1N spillover calibration

**State:** completed and terminally verified (`valid` evidence)

## Approved question

Under a matched short Pythia-14M pretraining budget, does stronger naive L1
pressure at the FFN hidden activation `h` increase its near-zero mass while the
untargeted attention operands `q_post`, `k_post`, and `v` lose near-zero mass?
Does this relationship differ between stock GeLU and ReLU at `h`?

## Conditions and matched controls

The ten conditions are two no-pressure activation controls plus
`lambda in {0.1, 0.5, 1, 5}` for each activation. `L1N` executes as the
operational `l1_naive` objective: task loss plus `lambda` times the unweighted
mean of the per-layer mean absolute `h` activations. Pressure sites are exactly
`[h]`.

- GeLU conditions use `A0`, whose unselected `h` operator is stock GeLU.
- ReLU conditions use `A1-H` with the standard ReLU operator at `h`.
- Controls use pressure method `none`, not a numerically special L1 weight.
- Conditions run serially as GeLU/ReLU pairs: control first, then ascending
  pressure weight.

Everything else is matched: pinned architecture and data revisions, random
initialization and its parameter hash, seeds, realized block schedule, global
batch, optimizer, LR schedule, complete validation, diagnostics, and retention.

## Scientific and operational definition

- Randomly initialize the pinned Pythia-14M config; never load released weights.
- MiniPile sequence length 2,048; model and data-order seeds are both `0`.
- Peak LR `4e-3`; AdamW `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`,
  global clip norm `1.0`, BF16 autocast, and FP32 parameters/state.
- Global batch 64, realized as microbatch 4 x accumulation 16 for every
  condition.
- One calibrated common step count of 451 gives every condition 59,113,472
  training input tokens. The conservative launch-through-figure p90 ETC is
  6,888.70 seconds (1h54m48.7s), inside the approved 7,200-second target.
- Warm up for `ceil(0.01 * max_steps)` updates, then cosine decay to 10% of peak.

## Validation, diagnostics, and figure estimand

Each condition evaluates all 500 validation documents after update 1 and from
the reloaded final checkpoint: 338 complete blocks, 692,224 input tokens, and
the 1,444-token excluded tail. The final pass records count-first exact-zero and
near-zero statistics at epsilon `1e-3` and `1e-2`, plus RMS/L2 moments for `h`,
`q_post`, `k_post`, and `v` in all six layers. It also records weight statistics
and retains every final checkpoint. Pressure conditions log separate task and
pressure gradient norms, dot product, cosine, conflict flag, pressure loss, and
combined-gradient clipping at every optimizer boundary.

The requested figure uses epsilon `1e-3`. Its x-coordinate is the count-pooled
`h` near-zero fraction. Its y-coordinate is the arithmetic mean of the three
separately count-pooled `q_post`, `k_post`, and `v` fractions. The three sites
have matched element counts, so this also equals pooling their integer hits and
denominators jointly. Each activation line contains control, 0.1, 0.5, 1, and
5 in pressure-strength order; connectors are guides, not fitted models.

## Manuscript relationship and interpretation

This directly probes the introduction's proposed sparsity-spillover pattern and
executes Eqs. `l1-pressure` and `naive-objective` from the methodology. A
down-right within-activation trajectory supports the simple signature; stable
or increasing attention near-zero mass refutes it. If pressure fails to raise
`h` near-zero mass, the intended manipulation was not achieved. Between-line
differences intentionally include the GeLU-versus-ReLU operator change.

With one seed, a local two-hour horizon, and one batch size, the run cannot
establish a general LR/pressure optimum, a causal propagation route, or a
manuscript-ready claim. Logical-product counters and clipping frontiers are
omitted; no runtime speedup is inferred.

## Approval and monitoring

- Design approved by the user on 2026-08-29 with `L1N = l1_naive`, figure
  epsilon `1e-3`, and an inclusive two-hour post-launch envelope.
- The user explicitly pre-approved local launch after successful implementation,
  verification, and calibration; no additional launch confirmation is required.
- The detached process writes durable per-step events. Read-only status is
  checked around 20%, 40%, 60%, 80%, and terminal completion rather than at a
  per-minute cadence.

## Implementation, verification, and calibrated launch

- Run-local scripts implement calibration, ten-condition execution, terminal
  verification, numerical reduction, PDF generation, and milestone monitoring.
- Six Run 002 focused tests and all 60 bootstrap tests pass. All nine Python
  files compile and `git diff --check` passes.
- Two production-shaped calibration samples exercised GeLU/ReLU controls and
  both `lambda=5` pressure paths. The locked repeat is
  `prelaunch/calibration-20260829-120142.json`; all losses and gradient metrics
  were finite, all 24 diagnostic rows were present, and checkpoint save/hash/
  reload succeeded.
- Peak allocated/reserved Torch memory was 6,450,855,424 / 8,306,819,072 bytes
  on the 12,820,480,000-byte RTX 5070 Ti Laptop GPU. Conservatively adding the
  preflight desktop/WDDM allocation leaves roughly 1.8 GiB headroom.
- The final seed-0 schedule contains 28,864 distinct complete blocks with no
  wrap and SHA-256 `4ee14c05906046815967e9e77c73df9acc41f44d96438fcef651cb40b7ea9014`.
- Exact code/config/schedule identities, execution/storage details, warnings,
  and the 6,888.70-second p90 envelope are locked in
  `prelaunch/launch-plan.json`.
- Monitoring uses a detached PowerShell process with `Start-Sleep` intervals of
  1,378 seconds, corresponding to about 20% of the expected p90 duration.

## Terminal result and artifact audit

- The detached cohort ran from `2026-08-29T12:04:52Z` through
  `2026-08-29T13:55:29Z`. Launch-through-verification-and-figure wall time was
  6,637.13 seconds (1h50m37.1s), within the approved two-hour envelope.
- All ten immutable attempts completed 451 optimizer steps and 59,113,472
  training input tokens each: 4,510 steps and 591,134,720 tokens in total.
- Terminal verification reports 20 complete validation passes, ten retained
  and hash-verified final checkpoints, identical initial-parameter, schedule,
  and run-code hashes, and non-null Git identity in every manifest. The Git
  dirty flag is disclosed because the approved run was launched from the
  working tree.
- Five detached monitor snapshots cover the planned 20%, 40%, 60%, 80%, and
  100% milestones. The terminal snapshot found the training process stopped,
  the GPU released, and the figure complete. Cohort stderr contains no OOM,
  traceback, non-finite, runtime, or explicit error match.
- Numerical values, estimands, and caveats are recorded in
  `observations/O001-h-vs-attention-near-zero.md`; the plotted reduction is
  `artifacts/figure_data.json`, and the publication output is
  `figures/01-h-vs-attention-near-zero.pdf`.

## Additional sitewise post-hoc figure

Figure 02 separates the attention average into `q_post`, `k_post`, and `v`
panels and adds `m`, with count-pooled `h` near-zero mass on every x-axis. The
original terminal artifacts already contain `h`, `q_post`, `k_post`, and `v`.
Because `m` was not selected before launch, `06_plot_sitewise.py` reloaded all
ten retained, hash-verified final checkpoints and measured operational `m` over
the same full 338-block validation coverage. This is a checkpoint-reconstructible
post-hoc activation diagnostic; it does not reconstruct training-time gradient
interaction.

The `m` pass completed in 23.06 seconds, produced six reconciled layer rows per
condition, and reproduced all ten stored final validation losses exactly. Its
count artifact is `artifacts/posthoc-m-activation-statistics.json`; the four-panel
reduction is `artifacts/figure_data_sitewise.json`; the observation is
`observations/O002-h-vs-site-near-zero-grid.md`; and the visually verified PDF is
`figures/02-h-vs-site-near-zero-grid.pdf`.

## Attention-output post-hoc figure

Figure 03 compares count-pooled `h` near-zero mass with the attention-branch
output immediately after `W_o` and before residual addition. The metric was not
stored at launch, so `08_plot_attention_output.py` reloaded the ten retained,
hash-verified checkpoints and reran complete validation. It captured both each
layer's `attention.dense` output and the corresponding
`post_attention_dropout` output entering the parallel residual sum. Dropout is
zero, and all 5,100 layer/batch comparisons were exactly equal.

The pass completed in 23.47 seconds, produced six reconciled layer rows per
condition, and reproduced all ten final validation losses exactly. Attention-
output near-zero mass increased from control through `lambda=1` for both
activations, then fell at `lambda=5`. The count artifact is
`artifacts/posthoc-attention-output-activation-statistics.json`; the plotted
reduction is `artifacts/figure_data_attention_output.json`; the observation is
`observations/O003-h-vs-attention-output-near-zero.md`; and the visually verified
PDF is `figures/03-h-vs-attention-output-near-zero.pdf`.

## Epsilon-1e-2 figure counterparts

Figures 04--06 reproduce Figures 01--03 with near-zero mass defined as
`abs(x) <= 1e-2`. No checkpoint replay was needed: every coordinate was reduced
again from the stored integer threshold hits and denominators over the same full
338-block validation coverage. The original terminal diagnostic supplies `h`,
`q_post`, `k_post`, and `v`; the verified post-hoc diagnostics supply `m` and
the post-`W_o`, pre-residual attention output.

- Figure 04 and O004: `figures/04-h-vs-attention-near-zero-eps1e-2.pdf` and
  `observations/O004-h-vs-attention-near-zero-eps1e-2.md`.
- Figure 05 and O005: `figures/05-h-vs-site-near-zero-grid-eps1e-2.pdf` and
  `observations/O005-h-vs-site-near-zero-grid-eps1e-2.md`.
- Figure 06 and O006: `figures/06-h-vs-attention-output-near-zero-eps1e-2.pdf`
  and `observations/O006-h-vs-attention-output-near-zero-eps1e-2.md`.

Their count-preserving reductions are
`artifacts/figure_data_attention_mean_eps1e-2.json`,
`artifacts/figure_data_sitewise_eps1e-2.json`, and
`artifacts/figure_data_attention_output_eps1e-2.json`. The PDFs were rendered
and inspected after annotation-layout checks.

## Candidate findings

Not promoted to `research/findings/` pending user review. Descriptively, L1N
raised `h` near-zero mass for both activations, but the requested simple
down-right spillover signature was not monotone: attention-site near-zero mass
rose through intermediate weights and fell at `lambda=5`. The strongest weight
also worsened final validation loss for both activations. This is a one-seed,
short-horizon result.
