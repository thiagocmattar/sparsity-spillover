# Signed activation distributions under train-time interventions

## Question

How do the complete A4-OL1 and A7-OL1 recipes reshape signed activation
distributions relative to A0 as the trained threshold increases?

## Method and coverage

Run 031 re-evaluates the same seven Pythia-14M step-712 checkpoints used in
Figure 05: A0 from Run 004, A4-OL1 from Run 015 and A7-OL1 from Run 014, with
kappa=0,.05,.5 for the latter two families. All share source seed 1234. Each
evaluation uses all 500 MiniPile validation documents packed into 338 complete
2,048-token blocks, with 692,224 input tokens and the excluded 1,444-token tail.
Weights and saved gates are frozen; no new training or post-hoc clipping is used.
Evaluation uses FP32 parameters, FP16 CUDA autocast and eager uncached attention.

Capture post-gate h,m,q_post,k_post,v at all six layers. Pool integer counts
across layers, blocks and sites before dividing. FFN activations combine h,m
with 80%/20% element weights; attention combines post-RoPE q,k and v equally.
Each model contributes 2,658,140,160 FFN and 1,594,884,096 attention elements.
These correlated elements are not independent experimental replicates.

The retained histogram has nominal .001 bins over [-8,8], separate tails and
extrema, and a separate exact-zero point mass. Display bins sum ten adjacent
native bins. Density is count/(all captured elements * actual bin width): its
integral is the nonzero probability in range, not one. The zero mass is listed
in each panel and is never spread into a KDE peak. Signed x remains linear.
The density axis is symlog, linear below .01, so large central peaks, lower
densities and genuinely empty gate regions can all be seen.

## Figure and caption

[Figure 05-v3](../figures/05-v3-activation-density-grid.pdf) is a separate
alternative; the original Figure 05 and Figure 05-v2 remain available.

**Signed activation distributions under train-time interventions (Pythia-14M).**
Columns pool FFN (h,m) and attention (q,k,v) activation elements; rows compare
trained thresholds kappa=0,.05,.5. Gray, blue and orange outlines and translucent
fills show A0, A4-OL1 and A7-OL1 histogram densities. The same A0 reference is
repeated across rows. Exact-zero probabilities are listed separately. Dashed
vertical lines indicate +kappa for A4/A7 FFN gates and +/-kappa for A7 attention
gates; A4 does not directly gate q,k,v. Density uses a symlog scale, while signed
activation x is linear. The common per-column x ranges show the central
distributions; all counts outside the view remain in the released histograms.
The plotted density is not renormalized to the visible range or to nonzero mass.

## Result

All seven full-validation evaluations passed. The largest loss discrepancy
from the retained eager endpoints is 0.0000688543 (tolerance 0.0005). The largest
site-level exact-zero discrepancy is 0.00336 percentage points; maximum RMS
discrepancy is 0.00004186. All count partitions and group reductions pass.

At kappa=.5, A4-OL1 places 99.31% of pooled FFN elements exactly at zero,
versus 93.64% for A7-OL1. Both leave nonzero FFN density above the positive
threshold. In attention, A4-OL1 produces a sharp central nonzero peak with
only 0.22% exact zeros, while A7-OL1 places 95.60% at zero and leaves nonzero
density outside the symmetric threshold gap. Thus small nonzero values and
exact-zero mass are visibly different outcomes of the complete recipes.

| Trained kappa | A4-OL1 FFN zero mass | A7-OL1 FFN zero mass | A4-OL1 attention zero mass | A7-OL1 attention zero mass |
| --- | ---: | ---: | ---: | ---: |
| 0 | 67.22% | 64.52% | <0.01% | <0.01% |
| .05 | 86.71% | 75.29% | <0.01% | 7.26% |
| .5 | 99.31% | 93.64% | 0.22% | 95.60% |

The FFN view [-2.5,3] excludes at most 0.1113% of elements; the attention
view [-4,4] excludes at most 6.6719% (A7-OL1, kappa=.05). The full native
[-8,8] histogram excludes at most 0.7607%, retained in explicit tail counts.
No displayed density is rescaled to conceal this missing mass. The shared
central views make the gate regions legible while the complete per-site/layer
histograms remain available for tail inspection.

## Caveats

This compares complete trained recipes, including their gates and OL1 pressure
histories. It does not isolate a causal pressure effect, show a training
trajectory, or establish a runtime benefit. Pooling can hide site-level changes;
the release retains each layer/site histogram. There is one seed per condition,
and the repeated A0 panels are not new evaluations or replicates. At FP16 gate
boundaries, a retained value can occupy the native bin straddling the nominal
kappa; that bin alone is not evidence of gate leakage. Logarithmic density
spacing means visual filled area is not a probability readout; probabilities
come from the retained integer counts.

## Source and reproduction

- Measurement: `runs/031-2026-09-08-signed-activation-density/02_evaluate.py`.
- Figure: `02_activation_density_v3.py` in this analysis.
- Source histograms: Run 031 `results/histograms/*.json.gz`.
- Plot provenance, exact zero mass and display-tail coverage:
  `activation-density-v3-data.json` in this analysis.

```powershell
.venv/Scripts/python.exe analyses/018-2026-09-08-results-materials/02_activation_density_v3.py
```
