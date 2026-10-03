# O001: Bilateral versus unilateral after LayerNorm

## Question and method

Does preserving large negative a/m values improve final quality under matched T4/Ph and T4/Pall pressure? Eight fresh 14M conditions, identical step-zero bytes, data schedule and 712-update recipe, kappa 0/.1; only the a/m sign rule changes within a pair. h/z stay unilateral.

## Coverage and result

All 500 validation documents, 338 complete 2,048-token blocks, 692,224 input tokens, 691,886 prediction tokens; 1,444-token tail excluded. Pooled integer activation, tile and product counts precede division.

Bilateral a/m has lower loss in 2/4 matched pairs. PH: 2/2 pairs favor bilateral. PALL: 0/2 pairs favor bilateral. Support depends on the pressure recipe; the simple claim that bilateral a/m improves quality across both recipes is not supported by this grid.

- PH, kappa=0: loss delta -0.085038, R_model delta -3.7852 percentage points.
- PH, kappa=0.1: loss delta -0.054435, R_model delta -3.6034 percentage points.
- PALL, kappa=0: loss delta +0.018569, R_model delta -3.8487 percentage points.
- PALL, kappa=0.1: loss delta +0.126668, R_model delta -1.3589 percentage points.

Under Ph, bilateral improves quality at both cutoffs while reducing model-wide logical opportunity. This is a quality/sparsity tradeoff, rather than an improvement on both axes.

Under Pall, unilateral has both lower loss and higher logical opportunity at each tested cutoff. The bilateral points are dominated on these two measured axes in this cohort.

## Distribution and structural diagnosis

Unilateral PH removes 48.56--50.78% of its own pre-gate squared energy from negative a/m values at or above the cutoff magnitude. Unilateral PALL removes 93.40--94.71% of its own pre-gate squared energy from negative a/m values at or above the cutoff magnitude. Bilateral removes none of that large-negative energy, as confirmed by the saved count/energy audit. These are different learned distributions; the removed energy is not a causal estimate of useful information.

- PH, unilateral: pre-gate a/m RMS 1.004--1.024; negative mass 49.40--50.88%.
- PH, bilateral: pre-gate a/m RMS 0.984--1.015; negative mass 49.77--49.79%.
- PALL, unilateral: pre-gate a/m RMS 0.969--1.021; negative mass 39.04--89.05%.
- PALL, bilateral: pre-gate a/m RMS 0.438--0.800; negative mass 50.01--52.53%.

Ph ends with a/m RMS near one and mean near zero, whereas the final distributions differ materially under Pall. Preserving large negative values is insufficient to predict quality under Pall. The RMS/affine and gradient summaries are descriptive diagnostics, not evidence that one specific pressure mechanism caused the loss change.

All four Ph conditions have zero completely empty 8x16 tiles at a and m, despite high elementwise zero rates in the unilateral conditions.

At Pall kappa=.1, a empty 8x16 tiles fall from 58.1223% (unilateral) to 8.3387% (bilateral); m falls from 0.010449% to 0.000000%. Tile pressure remains a separate deferred investigation.

## Figures and legend

1. `figures/01-paired-quality.pdf`: final validation loss; panels are pressure recipes; orange unilateral, blue bilateral. Only measured kappa endpoints are connected.
2. `figures/02-negative-energy.pdf`: squared pre-gate energy removed from negative values with magnitude at least the realized gate cutoff, separately a/m. Each denominator is that condition's own pre-gate energy, not a common activation distribution.
3. `figures/03-training-curves.pdf`: per-update global-batch task loss; solid kappa 0, dashed kappa .1; colors indicate the a/m rule.
4. `figures/04-am-tile-occupancy.pdf`: percentage of completely empty aligned 8x16 tiles in a/m, within each sequence, without padding. Orange unilateral, blue bilateral. The m panel uses a zoomed percentage scale to show its very rare empty tiles; labels display the actual percentages.

## Caveats and interpretation

One seed, two thresholds, a/m changed jointly. Equal cutoff does not match zero rates or rejected energy. Learned LayerNorm affine outputs need not stay centered at zero or within [-1,1]. Zero mean also does not imply zero median or exactly 50% negative mass. Gate-mask comparisons use each tensor's actual dtype, including the FP16 realization of kappa. Negative-energy rejection differences are descriptive and cannot by themselves prove mediation of the loss change. Pall applies L1 to post-gate a/m/h/z, so preserving negative a/m values also changes the values and gradients entering its pressure calculation; this comparison estimates the complete gate-rule change under the declared recipe. R_model and tile occupancy are logical/structural opportunities, not measured hardware gains. Pressure/gates use elementwise rules; tile pressure remains deferred. No automatic manuscript update.

## Provenance

Source script: `21_analyze.py`; source hashes and exact count exports: `results/results.json`. Complete numerical tables: `results/tables.md`. All final checkpoints and boundary diagnostics were retained and hash-verified locally before compute teardown.
