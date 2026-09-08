# Figure captions

Draft captions for five selected results figures: four training figures and
the kernel case study, 8 September 2026. Numbers below identify analysis files,
not final manuscript numbering.
Placement and intended claims are in [MANUSCRIPT-PLAN.md](MANUSCRIPT-PLAN.md).
The existing PDFs are unchanged. Replace descriptive appendix references with
LaTeX labels when inserting these captions into the draft.

## Figure 01: Quality-sparsity overview

**Quality-sparsity trade-offs from training interventions and post-hoc clipping.**
Validation loss versus model-wide logical sparsity
$\mathcal{S}_{\mathrm{model}}$ for Pythia-14M. Filled markers show 16 separately
trained checkpoints: A0, A1-H, and the A1-H-OL1, A4-OL1 and A7-OL1 sweeps.
Local OL1 varies $\lambda\in\{0.05,0.1,0.5,1\}$; A4/A7-OL1 vary
$\kappa\in\{0,0.01,0.05,0.1,0.5\}$ with fixed pressure weight $\lambda=1$
and trust budget 1. The gray dashed path with open markers applies
evaluation-only magnitude clipping at $a,m,h,z$ to A0, with calibration
targets $p=0,0.1,\ldots,0.9$. Six of its ten evaluations lie inside the
displayed loss range; the appendix supplies the full-range view. Connections
order evaluated settings and do not represent fitted Pareto envelopes or
intermediate trained models. The same seed, initialization, data order and
712-update budget are used for the trained conditions. Loss and sparsity use
all 338 complete 2,048-token blocks from 500 MiniPile validation documents,
excluding the 1,444-token tail. Logical sparsity counts zero-operand products,
not measured runtime savings.

Figure: [01-14m-overview.pdf](figures/01-14m-overview.pdf).
Evidence and display coverage: [O001](observations/O001-overview.md).
Generating code: [plots.py](plots.py), `overview`, invoked by [01_build.py](01_build.py).
All trained and clipped coordinates: [overview series](tables/overview-series.md)
and [Run 030 release](../../runs/030-2026-09-08-all-models-posthoc-clipping/results/README.md).

## Figure 02: Paired intervention effects

**Paired intervention effects.** Changes in validation loss (top; negative is
better) and model-wide logical sparsity $\mathcal{S}_{\mathrm{model}}$ (bottom;
percentage points) for 29 comparisons of separately pretrained Pythia-14M
conditions. Each delta is treatment minus reference, as specified in the
**Paired:** row; zero denotes no change. The **Intervention:** row describes
the operation and affected sites. Numeric labels give pressure weight
$\lambda$ or gate threshold $\kappa$; the dash marks the ReLU replacement,
which has no swept parameter. Parameter values increase from left to right
with ordinal spacing.
L1-to-OL1 comparisons share $\lambda$; pressure additions to A4/A7 and the
Q/K/V gate addition share $\kappa$. The remaining references are fixed
controls: A0 uses GELU and A1-H uses ReLU. One-sided gates act at $a,m,h,z$;
symmetric Q/K gates follow RoPE. Four-site and seven-site orthogonal L1 (OL1)
use fixed pressure weight $\lambda=1$ and trust budget 1. Pairs share
initialization, data order and training budget. All conditions use one seed
and the full MiniPile validation set: 500 documents packed into 338 complete
2,048-token blocks, excluding the 1,444-token tail. These are descriptive
paired comparisons, not successive training updates; logical sparsity does
not measure runtime speedup.

Figure: [02-blocked-intervention-effects.pdf](figures/02-blocked-intervention-effects.pdf).
Evidence, reference map and caveats: [O002](observations/O002-blocked-effects.md).
Generating code: [plots.py](plots.py), `effects`, invoked by [01_build.py](01_build.py).
All 29 contrasts: [blocked effects](tables/blocked-effects.md).

## Figure 03: Transfer across model sizes

**Quality-sparsity trade-offs across Pythia model sizes.** Columns compare
14M, 70M and 410M. Solid curves connect separately trained A4-OL1 and A7-OL1
checkpoints at $\kappa=0,0.01,0.05,0.1,0.5$; dotted open-marker curves show
A0 post-hoc clipping at $a,m,h,z$ for targets $p=0,0.1,\ldots,0.9$.
The 60 evaluations appear in both rows. Top: validation loss versus
$100\mathcal{S}_{\mathrm{model}}$; vertical guides indicate A4/clipping-site
and A7 analytic reach ceilings. Bottom: loss minus unmodified same-size A0
versus $100U_{\mathrm{arch}}$, where
$U_{\mathrm{arch}}=\mathcal{S}_{\mathrm{model}}/
\mathcal{S}_{\mathrm{model}}^{\max}(\mathrm{A7})$ for **every curve**.
The A7 reference ceilings are 29.9524%, 49.4239% and 87.2452%; the A4 guide is
not the normalization denominator. At $\kappa=0.5$, A7-OL1 has lower loss and
greater sparsity than A4-OL1 at all three sizes; the ordering at smaller
thresholds is not uniform. Each size uses one seed, matched conditions and
the same training-token budget, with a lower peak learning rate at 410M.
Every evaluation covers 338 complete validation blocks from 500 documents,
excluding the 1,444-token tail. This is transfer of a complete-recipe
comparison, not an isolated pressure effect or a scaling law.

Figure: [03-scale-transfer-and-ceilings.pdf](figures/03-scale-transfer-and-ceilings.pdf).
Evidence and normalization: [O003](observations/O003-scale-transfer.md).
Generating code: [plots.py](plots.py), `scaling`, invoked by [01_build.py](01_build.py).
Numerical contrasts: [scale-paired recipes](tables/scale-paired-recipes.md).
That table's existing `U` columns use recipe-specific ceilings; they must not
be copied as this figure's common-A7 normalization.

## Figure 05-v3: Activation distribution reshaping

**How interventions reshape activation distributions (Pythia-14M).**
Columns pool FFN activations $(h,m)$ and attention activations $(q,k,v)$;
rows compare trained thresholds $\kappa=0,0.05,0.5$. Gray, blue and orange
lines with translucent fills show A0, A4-OL1 and A7-OL1. Integer histogram
counts are pooled across all layers and validation blocks before division;
FFN pooling weights $h:m$ by element counts (80:20), while $q,k,v$ contribute
equally. Signed activation $x$ is linear; density uses a symlog scale, linear
below 0.01. **Exact zeros are excluded from the curves and tabulated in the
appendix.** Density divides nonzero bin counts by all captured elements and
bin width; it is not renormalized to nonzero mass or the visible range.
Dashed lines mark $+\kappa$ for A4/A7 FFN gates and $\pm\kappa$ for A7
attention gates, with Q/K measured after RoPE. A4 does not directly gate or
pressure $q,k,v$; A7 does both. The same A0 reference is repeated across
rows. Seven frozen checkpoints each use all 338 complete blocks from 500
validation documents, excluding the 1,444-token tail. These are trained
distributions with no additional post-hoc clipping. Complete histograms and
tail counts accompany the figure; recipe differences do not isolate a
causal pressure mechanism.

Figure: [05-v3-activation-density-grid.pdf](figures/05-v3-activation-density-grid.pdf).
Evidence, exact-zero table and tail coverage: [O011](observations/O011-activation-density-v3.md).
Generating code: [02_activation_density_v3.py](02_activation_density_v3.py).
Display provenance: [activation-density-v3-data.json](activation-density-v3-data.json).
Source measurement and retained histograms:
[Run 031 release](../../runs/031-2026-09-08-signed-activation-density/results/README.md).

## Figure 07: Kernel realization

**Sparse kernel auto-research for full-model acceleration.** (a) Best
numerically qualified incumbent on the fixed Pythia-14M A7-OL1,
$\kappa=0.5$ checkpoint across 42 eligible kernel iterations, initialized at
native $1\times$ and reaching $1.7830\times$. (b) Final selected kernel K050
on the 30 included checkpoints; colors and markers identify trained recipes.
The dashed line is an unweighted descriptive OLS fit with an intercept
($R^2=0.8167$); dotted horizontal lines mark native $1\times$.
Speedup is measured against each checkpoint's native PyTorch/SDPA CUDA-graph
baseline on one RTX5090, using BF16, batch one, and uncached 2,048-token
inference with full vocabulary logits. Timings use 64 fixed inputs, seven
paired passes and three fresh processes. All 30 checkpoints qualify
numerically over 338 complete validation blocks from 500 MiniPile documents,
excluding the 1,444-token tail. Their geometric mean speedup is
$1.2340\times$. Model-wide sparsity uses the canonical FP16 measurement
shared with the training figures; both speedup axes use a restricted range.
Checkpoint quality, weights and topology vary, so the association does not
isolate a causal sparsity effect or establish equal-quality acceleration.
The appendix reports fusion and sparse-path ablations.

Figure: [07-kernel-realization.pdf](figures/07-kernel-realization.pdf).
Evidence, runtime protocol and caveats: [O007](observations/O007-kernel-realization.md).
Generating code: [plots.py](plots.py), `kernels`, invoked by [01_build.py](01_build.py).
Numerical source: [figure_data.json](figure_data.json), `runtime`, reduced by
`evidence.py:runtime_subset`; [runtime summary](tables/runtime-summary.md).
Use this 30-checkpoint caption and cohort statistics together when replacing
the draft's older 35-checkpoint Run 029 figure.
