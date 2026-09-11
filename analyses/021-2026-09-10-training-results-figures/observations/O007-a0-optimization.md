# A0 training loss and gradient norm across sizes

## Question and scope

How do untreated A0 training loss and pre-clipping global gradient norms evolve
over the common training-token budget for Pythia-14M, 70M and 410M?

The author requested this two-panel appendix diagnostic on 11 September 2026.
It is generated in Analysis 021 only; no manuscript files are changed.

## Sources and coverage

The A0 identities are selected from Analysis 018
[figure_data.json](../../018-2026-09-08-results-materials/figure_data.json), exactly
matching the controls used by Figure 05. The training events and manifests are:

- 14M: Run 004, `artifacts/attempts/001-20260829-221007-bb5288c8/`.
- 70M: Run 018, `artifacts/attempts/001-20260901-133016-4e43b254/`.
- 410M: Run 019, `artifacts/attempts/001-20260902-141527-bcb97fb1/`.

Each `events.jsonl` contains 712 training records, giving 2,136 records total.
Each boundary consumes 2,097,152 input tokens, ending at 1,493,172,224 tokens
per size. There are no missing steps, nonfinite measurements, overflows or
skipped A0 updates. Each size has one seed, 1234. Exact event/manifest paths,
SHA-256 hashes, initialization and schedule identities, optimizer settings,
raw records and displayed smoothing values are retained in the
[reduction](../data/a0-optimization.json).

These are training-batch measurements, not repeated full-validation losses.
The A0 final validation values in Figure 05 use all 500 validation documents,
338 complete 2,048-token sequences, with the 1,444-token tail excluded.

## Metric definitions and display

- Horizontal coordinate: logged `input_tokens_seen / 1e9`, without extrapolated
  records at zero tokens. Both panels use the range 0 to 1.5 billion tokens.
- Training loss: logged `task_loss`, the arithmetic mean causal-language-model
  loss over equally sized microbatches in each accumulated optimizer boundary.
- Gradient norm: logged `adamw_gradient_norm_pre_clip`, the global L2 norm of
  accumulated task gradients over trainable parameters, after dynamic-loss
  unscaling and before clipping at norm 1.0. This is neither the post-clipping
  norm nor an AdamW-preconditioned update direction.

The definition is implemented in Run 004 `optimizer_boundary.py:run_recipe_boundary`
(also reused by the larger A0 controls) and
`src/sparsity_research/optimization.py:clip_adamw_gradients`. The boundary divides
loss-scaled gradients by their loss scale before computing the norm. PyTorch's
global L2 clipping returns the pre-clipping value; the post-clipping norm is
logged separately. Retained pre/post values reconcile with the implemented
threshold. Analysis 011's [original A0 diagnostic](../../011-2026-09-03-pythia14m-70m-410m-selected-ladder/observations/O003-a0-gradient-norm-vs-tokens.md)
records the same semantics and complete histories.

Both panels show faint raw trajectories and thicker centered **nine-step
arithmetic moving means**, with partial windows at the first/last four steps
and no padded values. The full window covers 18,874,368 input tokens, about
1.3% of the total budget. Gradient smoothing occurs in the original norm units,
before plotting on the logarithmic axis. All raw values remain visible.
The loss axis spans [3.5, 11.5], and the logarithmic norm axis spans [0.15, 35],
covering every raw and smoothed value. Blue, orange and purple denote 14M,
70M and 410M consistently across the two panels. One shared legend sits below.
The final polish shortens the titles to "A0 training loss" / "A0 gradient
norm", labels the norm axis "Pre-clipping global task-gradient norm", and
reduces raw-trace opacity to 0.12 while keeping the smoothed curves dominant.
The longer norm label wraps across two lines for legibility.

## Figure and proposed caption

[Publication PDF](../figures/06-a0-optimization.pdf)

**A0 optimization trajectories across model sizes at a common token budget.**
(a) Training loss versus cumulative input tokens for Pythia-14M, 70M and 410M.
(b) Global task-gradient L2 norm after loss unscaling and accumulation but before
clipping at 1.0, shown on a logarithmic axis. Faint curves show all 712 logged
optimizer boundaries per size; thicker curves show a centered nine-step
arithmetic moving mean with partial windows at the edges. Each size trains on
1.493 billion input tokens using one seed. Colors identify the same model size
in both panels. Global norms are not normalized by parameter count. The 410M
run uses a lower peak learning rate than the smaller runs, so these diagnostics
describe the executed training protocols rather than isolating a model-size
effect or identifying an optimization defect.

## Results and limits

| Size | Final logged training loss | Final pre-clip norm | Maximum pre-clip norm | Clipped steps |
| --- | ---: | ---: | ---: | ---: |
| 14M | 5.2439 | 0.3035 | 2.0019 | 5/712 |
| 70M | 3.9830 | 0.3612 | 3.5132 | 8/712 |
| 410M | 4.4483 | 1.2628 | 26.9615 | 56/712 |

Training loss decreases across the common budget at every size. The 70M run
finishes with lower training loss than 410M; these training values are distinct
from final validation loss. The 410M pre-clipping norm rises during the later
part of training and finishes above the clip threshold, while the smaller
models finish below it.

These are unnormalized global norms over differently sized parameter vectors.
Their cross-size differences cannot be interpreted as per-parameter gradient
strength. The larger run uses a peak learning rate of 3e-4 rather than 1e-3;
equal input tokens also imply different tokens per parameter. Neither this
figure nor its smoothed traces establishes why 410M has its observed validation
loss. The curves are descriptive, single-seed training histories; smoothing is
a display aid and provides no uncertainty estimate.

## Reproduction and verification

Source script: [06_a0_optimization.py](../06_a0_optimization.py).

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/06_a0_optimization.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_a0_optimization.py -q
```

Three focused tests verify complete step/token coverage, the exact pre-clipping
metric and raw trajectories, source-reconciled final losses, and moving-mean
edge behavior. The 7.4-by-2.95-inch PDF was rendered and visually checked.
