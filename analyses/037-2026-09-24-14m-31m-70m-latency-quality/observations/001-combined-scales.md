# Three Pythia sizes: validation loss and full-model latency

## Question

Does a roughly 30M Pythia model, trained with the same Base and T2/Ph recipes
as the 14M/70M study, add useful points to the measured loss-latency frontier?
The approved set is Base plus kappa 0,0.01,0.05,0.1,0.5: six training conditions.

## Method and coverage

Run054 uses the pinned Pythia-31M architecture (30,494,720 parameters), random
seed-1234 small_init/wang_init, the same MiniPile cache and flattened example
order, and 712 updates of 1,024 sequences of 2,048 tokens per condition. Its
optimizer, learning-rate schedule, dynamic FP16 training and MB4/GAS256 match
the executed 70M recipe. T2/Ph means one-sided h/z gates retaining equality,
with h-only post-gate orthogonal L1 pressure (weight and step budget both one).
Base has GELU, no gates and no activation pressure. Six H200s ran the conditions
in parallel. A slow host was replaced using unchanged scientific inputs; the
interrupted attempt is retained and explicitly excluded from the completed set.

Quality is final-checkpoint FP16 validation over all 500 documents: 338 complete
2,048-token blocks, 692,224 input tokens and 691,886 next-token predictions.
The 1,444-token tail is excluded. Final activation, weight, logical-product,
gradient-interaction, clipping and recovery artifacts are retained.

The 31M kernel specializes Run045 opt073 for width 256, FFN 1024 and head width
32. It preserves that implementation's policy; no optimization search was run.
Timings use one RTX 5090, BF16, batch one, T=2048, full 50,304-token logits and
CUDA-graph replay. Each of three fresh processes uses 64 fixed validation inputs
and seven paired timing passes (448 samples per backend). The displayed value
is the geometric mean of the three process geometric means. Every process
qualified against native eager on all 338 blocks at the declared pointwise,
relative-L2 and pooled-loss tolerances. All 18 passed, using one GPU UUID and
one runtime source identity. Replicate one also retained complete BF16 activation
moments, row occupancy and independently checked issued/bypassed work counters.

The collector checks the complete training cohort, checkpoint-file hashes,
validation coverage, all raw timing reductions and runtime-diagnostic coverage.
The plot preserves all 68 original Analysis034 Figure02 execution coordinates
and adds seven: 75 executions from 72 checkpoints, including six Base executions.
Historical sessions and implementations remain unchanged, including the Run047
70M T2/Ph kappa=0.5 endpoint. Run051/052 measurements are not included.

## Legend and caption

**Validation loss versus full-model latency for Pythia-14M, Pythia-31M and
Pythia-70M.** Both axes are logarithmic. Circles, triangles and squares denote
14M, 31M and 70M. Blue highlights T2/Ph; orange highlights T7/Pall; other
historical recipes are gray. Hollow charcoal markers denote Base kernels and
smaller filled charcoal markers Base PyTorch. Vertical dotted guides mark each
Base validation loss. Lines connect threshold settings within an existing recipe
and size; they are not a fitted cross-size scaling law. Lower and left is better.
The near-coincident 14M Base markers retain their distinct measured coordinates.

## Results

| 31M condition | Validation loss | Kernel ms | PyTorch ms | R_model (%) |
|---|---:|---:|---:|---:|
| Base | 4.565514 | 1.106255 | 1.023861 | 0.000018 |
| T2/Ph, kappa=0 | 4.688476 | 1.020591 | 1.068473 | 8.588719 |
| T2/Ph, kappa=0.01 | 4.701284 | 1.017293 | 1.071115 | 8.732630 |
| T2/Ph, kappa=0.05 | 4.702921 | 0.969383 | 1.072476 | 9.056343 |
| T2/Ph, kappa=0.1 | 4.757208 | 0.908418 | 1.071218 | 9.259301 |
| T2/Ph, kappa=0.5 | 5.220418 | 0.681118 | 1.067923 | 9.465328 |

R_model is the pooled logical-product opportunity, not a measured speedup.
Its h/z analytic all-zero reach ceiling is 4,026,531,840 / 42,483,056,640
products per full input sequence (9.477971%). The tiny Base value reflects
natural numerical zeros, not an activation intervention.

The middle scale adds five **nominal measured Pareto points**: Base PyTorch and
the four T2/Ph settings from kappa=0 through 0.1. The complete plotted set has
16 nominal Pareto points. For example, 31M Base PyTorch (loss 4.565514,
1.023861 ms) dominates the former 70M T2/Ph kappa=0.5 point (loss 4.838034,
1.214368 ms). Within 31M, kappa=0.1 lowers latency by 11.28% relative to Base
PyTorch while increasing validation loss by 0.191694. This supplies intermediate
quality/latency choices between the small and large model families.

The trade-off is not uniformly favorable. The 31M Base kernel is slower than
Base PyTorch. The 31M kappa=0.5 point is dominated by smaller models: for example,
14M T2/Ph kappa=0.1 has lower loss (5.151098) and latency (0.573427 ms) than
31M kappa=0.5 (5.220418, 0.681118 ms).

## Caveats and interpretation limits

This is an empirical frontier under the displayed implementations and sessions.
It supports the user's intermediate-scale comparison, not a fitted scaling law.
There is one training seed, three model sizes, and architecture-specific kernels.
14M uses K050; 70M uses the historical opt073-era records; 31M uses the fixed
specialization. Hardware class and workload match, but historical sessions do
not share a GPU UUID. No timing rescaling or extrapolation is applied.

Nominal Pareto membership uses the recorded means without uncertainty margins.
The kappa=0 and 0.01 advantages over 31M Base PyTorch are only about 0.32% and
0.64%; they should not be treated as robust statistical separations. All raw
samples and the three per-process means are retained. For example, Base PyTorch
process means span 1.021292-1.025663 ms and kappa=0 kernel means span
1.018622-1.021700 ms. Larger changes should still be interpreted within this
fixed benchmark and the single-seed quality comparison. Logical opportunities,
BF16 runtime operand sparsity and issued kernel work are distinct measurements.

## Sources and verification

- Source run: `runs/054-2026-09-24-pythia31m-t2-ph/`, including
  `artifacts/verification.json`, per-attempt metrics and transfer inventories,
  and `latency/artifacts/final-001-*-grid.json` with their 18 process directories.
- Historical source: Analysis034 `data/combined-figure.json`, SHA-256
  `ad3d09de96dda3bb7d67d6e4aaf75bd593a52a42226a04a4febb5dcb71ed7954`.
- Generating scripts: `../01_collect.py` and `../02_plot.py`.
- Exact new values and evidence hashes: `../data/31m-results.json`.
- Combined points, nominal frontier, axis checks and output hash:
  `../data/combined-figure.json`.
- Figure: `../figures/01-14m-31m-70m-latency-quality.pdf`.

All source checks passed. The PDF was rendered with Poppler and visually checked
for clipping, overlap, legend readability and point coverage. The original PDF
remains unchanged. All agreed scientific artifacts were copied and verified
locally before terminating the corresponding Pods. The unused interrupted host's
missing package-freeze text is documented in the run README; all successful
endpoints retain their package freezes. No manuscript or canonical finding was
edited or promoted.
