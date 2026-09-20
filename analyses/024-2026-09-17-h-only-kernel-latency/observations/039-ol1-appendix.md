# Activation pressure: formulation, L1 comparison and realized OL1 geometry

Figure: [18-14m-70m-ol1-geometry.pdf](../figures/18-14m-70m-ol1-geometry.pdf).
Table: [t1-l1-ol1.tex](../data/t1-l1-ol1.tex).
Evidence: [ol1-appendix.json](../data/ol1-appendix.json).
Builder: [29_ol1_appendix.py](../29_ol1_appendix.py).

## Question and approved manuscript scope

Make Appendix `app:pressure` explain the implemented OL1 update before its
diagnostics. Add the 14M T1/P1 L1-versus-OL1 comparison at every retained
lambda and extend the earlier two-panel OL1 geometry view to all four
Figure 1 pressure recipes. Per the user's follow-up, the figure shows only
14M in two panels; its requested existing filename is retained. This reuses retained evidence;
no training, evaluation, timing or cloud work is launched.

## Method and coverage

The formula is checked against `src/sparsity_research/pressure.py`,
`optimization.py` and the executed run wrappers. Their recorded hashes match
the current files. Pressure is the equal-weight mean of site-layer mean
absolute activations after any gates. Ordinary L1 adds the weighted pressure
gradient to the task gradient before clipping and AdamW. OL1 accumulates
the two gradients separately at the same parameters and on the same
microbatches, clips the task gradient only, takes the task-only AdamW step,
then applies an independently preconditioned pressure correction. It uses the
newly updated task second moment, without pressure momentum; projection and
the relative norm cap pool all eligible parameters. The manuscript retains
the actual epsilon placement and projection guard. Geometry precedes group
learning rates and excludes decoupled weight decay.

For the comparison table, the eight Run004/009 endpoints share initialization,
seeds (1234), block order, optimizer settings and 712 training steps. Both
methods use ReLU at h and pressure only at h (T1/P1); lambda is .05, .1, .5
or 1, and OL1 uses b=1. Loss is the ordinary reloaded final-checkpoint pass,
not the slightly different logical-diagnostic loss. Model-wide sparsity
pools integer zero-product counts before division by the full-model count,
including the output head. Both metrics cover all 338 complete 2048-token
blocks from the 500 validation documents, with 1,444 tail tokens excluded.
The retained evaluation convention is FP16 autocast with FP32 parameters.

The evidence export retains Figure 1's 40 OL1 checkpoint identities:
T4/Ph, T4/Pall, T7/Ph and T7/Pall at both sizes and kappa=0,.01,.05,.1,.5.
The plotted subset is exactly the 20 14M checkpoints (14,240 records);
the ten excluded 70M pairs remain recoverable in the evidence export.
All 28,480 training records are present (712 ordered boundaries per condition).
All have lambda=b=1 and epsilon=1e-12. The reducer checks projection, cosines,
cap scaling, correction status and absence of overflow/skipped updates.
Run012 is classified by its already-audited realized h-only capture, despite
its historical four-site manifest declaration; its original wrapper hash
and the Analysis023 audit are retained. Other runs' captured tensor counts
match their declared pressure sites.

The left panel shows 100*rho_opp, where
`rho_opp=max(0,-dot(u,w))/(norm(u)^2+eps)`. This is removed pressure-component
norm divided by task-direction norm, before the cap; it is not a fraction
of the original pressure norm. All steps pass the task-norm projection guard.
Points and bars are medians and empirical 25th/75th percentiles over training
steps, with aligned steps contributing zero. All plotted 14M medians and IQR endpoints are positive, so the axis is
logarithmic. Aligned zeros remain in the samples used for the quantiles. Threshold positions
are categorical and equally spaced, with small offsets for readability.

The right panel uses a logarithmic r/b axis. Faint lines are individual threshold
trajectories; the bold line is their unsmoothed pointwise median. Cap activity
uses the actual `trust_scale < 1` condition, including numerical stabilization.
All 712 records are included, including each run's first zero-learning-rate
boundary. Cap activity there describes geometry, not a realized parameter move.

## Results

| Recipe | 14M cap-active steps | 14M median r/b | 70M cap-active steps | 70M median r/b |
| --- | ---: | ---: | ---: | ---: |
| T4/Ph | 9/3560 (0.25%) | 0.1020 | 36/3560 (1.01%) | 0.0445 |
| T4/Pall | 18/3560 (0.51%) | 0.2533 | 72/3560 (2.02%) | 0.2325 |
| T7/Ph | 9/3560 (0.25%) | 0.0860 | 39/3560 (1.10%) | 0.0391 |
| T7/Pall | 3453/3560 (96.99%) | 57.4851 | 3560/3560 (100%) | 25.7719 |

Seven-site all-site pressure usually saturates the norm budget, while h-only
and four-site all-site pressure rarely do. Thus lambda-invariance after
saturation does not make lambda irrelevant across all target sets.

| Lambda | L1 S_model (%) | L1 validation loss | OL1 S_model (%) | OL1 validation loss |
| ---: | ---: | ---: | ---: | ---: |
| .05 | 3.1416 | 5.2061 | 3.1452 | 5.1981 |
| .1 | 3.3363 | 5.1655 | 3.3386 | 5.1594 |
| .5 | 3.7876 | 5.1127 | 3.7680 | 5.1102 |
| 1 | 3.9493 | 5.1023 | 3.9384 | 5.1210 |

At T1/P1 the sparsity difference is below .02 percentage points at every
lambda. OL1 loss is lower at the first three weights and higher at lambda=1.
This is a single-seed, matched-endpoint comparison, not evidence of universal
quality superiority or a statistical significance claim.

## Figure caption and manuscript association

**OL1 geometry depends on pressure placement at 14M.**
**(a)** Median removed opposing component, 100*rho_opp; bars span the
25th--75th percentiles over 712 training steps per condition, not confidence
intervals. **(b)** Pre-cap norm ratio r/b: faint curves show individual
thresholds, and bold curves their pointwise median. The dotted line marks
the norm cap. Colors and line styles match Figure 1; both y-axes are
logarithmic, and lambda=b=1 throughout.

PDF copies in `manuscript/draft/figures/18-14m-70m-ol1-geometry.pdf` and the
generated table in `manuscript/draft/tables/t1-l1-ol1.tex` are byte-identical
to these analysis artifacts, with provenance in their SOURCES.json files.
Both are introduced in `methodology-appendix.tex`, `app:pressure`, after the
method formulation. The old 14M-only diagnostics subsection in
`results-appendix.tex` is consolidated here; its diagnostic and figure labels
remain valid. The main-method paragraph is minimally corrected to distinguish
ordinary L1 from the OL1 dual update and avoid claiming maximum-alignment
steps before the norm cap binds. No universal loss-preservation claim is made.

## Caveats

The stabilized projection is approximately orthogonal under conflict. It
constrains the correction relative to Adam's adaptive direction, not the raw
task gradient, and cannot guarantee task-loss or validation-loss preservation.
Target-set changes alter both objective composition and normalization.
Training-step intervals do not represent seed uncertainty; geometry alone
does not identify the cause of validation-loss differences. The lambda sweep
is at 14M T1/P1 only and does not establish multisite lambda sensitivity.

## Verification and reproduction

From the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/29_ol1_appendix.py
.venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_pressure.py analyses/021-2026-09-10-training-results-figures/test_ol1_geometry.py analyses/024-2026-09-17-h-only-kernel-latency/test_ol1_appendix.py
```

All 16 focused tests pass, covering pressure mathematics, historical geometry,
complete cohort membership, requested 14M-only panels, integer-count and loss
provenance, genuine zeros,
pooled cap fractions, source hashes and manuscript artifact identity. The
35-page manuscript compiles with resolved references/citations and no overfull
boxes. The new standalone figure and manuscript pages 12--13 were visually
checked, together with the revised main-method paragraph on pages 3--4.
