# Figure 07 v2 - Model-wide sparsity and projection-skipping gain

## Question

How does model-wide sparsity relate to the full-model speedup from enabling
projection skipping? The author requested a separate v2 of Figure 07 with
S_model replacing projection MMA bypass on panel (b)'s x-axis. The original
PDF and its data are preserved. The initial revisions were analysis-only;
[O010](O010-kernel-manuscript-adoption.md) records the subsequent approved
manuscript adoption without changing this artwork.

## Method, coverage and source

[07_kernel_realization_v2.py](../07_kernel_realization_v2.py) reuses the
original figure's [verified reduction and plotting code](../07_kernel_realization.py).
It joins canonical S_model to each of the same 30 projection-gain observations
by checkpoint identity. The previously computed unweighted OLS with an intercept
is retained in the data for reproducibility but is no longer displayed in panel (b).
Panel (a), the four legend groups, 55:45 layout and reference lines are retained.
Panel (b) uses the same 0–30% displayed sparsity range as panel (a).
The author's follow-up adds four connected threshold sweeps to panel (b):
four-/seven-site recipes, each with and without OL1. Each connects the five
separately trained thresholds 0, 0.01, 0.05, 0.1 and 0.5 in that order.
Solid curves/open markers denote no OL1; dashed curves/filled markers denote
OL1. A small panel-specific key defines that encoding. One κ = 0.5 label per
color marks the OL1 endpoint; no κ = 0 labels are shown. Panel (b) has no OLS
line or R² annotation. Two muted gray paths connect the one-site naive-L1
and OL1 sweeps separately, each starting at ReLU and continuing through
λ = 0.05, 0.1, 0.5 and 1. Their markers remain unchanged. Panel (a)'s markers
and annotations remain unchanged. The unconnected v2 is retained in commit
`7be29da` for rollback.

The gain remains the geometric-mean all-skips-off K050 latency divided by the
geometric-mean attention-dense/projection-on K050 latency. It is a full-model
timing ratio with the same fusion and attention skipping disabled in both
controls, not a projection-only timing or a native-normalized speedup ratio.

Sources and protocol are unchanged from [O008](O008-kernel-realization.md):
Analysis 018 canonical FP16 logical counts, Run 029's qualified 30-checkpoint
timings, and the [investigation](../investigation/README.md)'s reconstruction
of raw latency geometric means. Logical measurements cover all 338 complete
validation blocks from 500 documents, excluding the 1,444-token tail. BF16
timings use the original 64 inputs, seven paired passes and three fresh
processes per implementation on RTX5090, at batch one and sequence length
2,048 with full-vocabulary output. No new model execution is performed.

## Figure and caption

[Figure 07 v2](../figures/07-kernel-realization-v2.pdf).

**Model-wide sparsity relates differently to native-relative speedup and the
gain from projection skipping.** Both panels show the same 30 Pythia-14M
checkpoints against canonical model-wide sparsity. (a) Full K050 speedup
relative to each checkpoint's native PyTorch/SDPA implementation.
(b) Projection-skipping gain: all-skips-off latency divided by
projection-on/attention-dense latency, retaining identical fusion and disabled
attention skipping. The gray dashed line in panel (a) is a descriptive OLS fit
with an intercept (R² = 0.817); dotted lines mark 1×. Black squares identify A0,
gray circles identify nine 1-site checkpoints (ReLU and local L1/OL1), and
blue diamonds/orange triangles identify the ten four-site/seven-site
checkpoints. In panel (b), colored solid lines/open markers denote recipes
without OL1 and colored dashed lines/filled markers denote recipes with OL1.
Lines connect separately trained settings in increasing κ = 0, 0.01, 0.05,
0.1 and 0.5; they are not training trajectories or interpolated models.
Muted gray paths connect one-site naive-L1 (solid) and OL1 (dashed) settings
from ReLU through increasing λ = 0.05, 0.1, 0.5 and 1, with unchanged markers.
One κ = 0.5 label per color identifies the high-threshold multisite OL1 endpoints. The labeled
seven-site + OL1, κ = 0.5 endpoint in panel (a)
reaches 1.78× native-relative speedup. Canonical sparsity uses FP16 and full
validation coverage; timings use BF16 and the matched 64-input subset.
These associations across different checkpoints do not isolate causal effects.

## Result and limitations

The undisplayed panel-(b) fit has Pearson r = 0.704083, OLS R² = 0.495733; slope is
0.016199614 gain units per sparsity percentage point, with intercept 0.934545121.
Panel (a) remains R² = 0.816718. The previous panel (b), using projection MMA
bypass, has R² = 0.946105 on exactly the same projection-gain outcomes.
Model-wide sparsity counts scalar zero-product opportunities across all six
operation families, including attention; it is not projection structure or
work eliminated by the kernel. The new fit must not inherit the earlier
instruction-bypass interpretation. Native costs, recipe, threshold, weights
and quality vary across checkpoints, and neither fit is a causal estimate.

The high-threshold endpoints in panel (b) all have κ = 0.5:

| Recipe | Checkpoint | S_model (%) | Projection-skipping gain |
|---|---|---:|---:|
| 4-site | c15 | 10.215537 | 1.324139× |
| 4-site + OL1 | c20 | 12.713449 | 1.378251× |
| 7-site | c25 | 15.386813 | 1.316134× |
| 7-site + OL1 | c30 | 27.482684 | 1.325269× |

These are the upper endpoints of the four connected curves. The gain compares
implementations of the same checkpoint; connecting the curves does not turn
the between-checkpoint threshold changes into matched runtime ablations.

At κ = 0.5, adding OL1 raises S_model by 2.498 pp for four-site recipes and
12.096 pp for seven-site recipes. Projection-skipping gain rises from 1.324×
to 1.378× and from 1.316× to 1.325×, respectively. The seven-site sparsity
increment thus accompanies little additional projection-skipping benefit;
this does not establish that OL1 adds no useful sparsity, since S_model also
counts attention products and the four-site projection gain increases.

## Reproduction and verification

[kernel-realization-v2.json](../data/kernel-realization-v2.json) retains source
hashes, all qualified comparisons, the 30 matched panel-(b) points, both
regressions, four ordered threshold sweeps, two ordered local pressure-weight
sweeps and the original PDF's SHA-256.
The original Figure 07 is untouched.

```powershell
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/07_kernel_realization_v2.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_kernel_realization.py analyses/021-2026-09-10-training-results-figures/test_kernel_realization_v2.py -q
```

Three v2 tests check checkpoint/count joins, unchanged gains and original PDF,
independent regression, all plotted x/y values, and the four ordered sweeps
with the correct pressure encodings. The three original tests also pass.
The PDF was rendered and visually inspected.
