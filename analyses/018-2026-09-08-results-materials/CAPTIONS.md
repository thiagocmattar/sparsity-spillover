# Figure captions

## Figure 02: Paired intervention effects

**Paired intervention effects.** Changes in validation loss (top; negative is
better) and model-wide logical sparsity $\mathcal{S}_{\mathrm{model}}$ (bottom;
percentage points) for 29 comparisons of separately pretrained Pythia-14M
conditions. Each delta is treatment minus reference, as specified in the
**Paired:** row; zero denotes no change. The **Intervention:** row describes
the operation and affected sites. Numeric labels give pressure weight
$\lambda$ or gate threshold $\kappa$; the dash marks the dose-free ReLU
replacement. Doses increase from left to right with ordinal spacing.
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
