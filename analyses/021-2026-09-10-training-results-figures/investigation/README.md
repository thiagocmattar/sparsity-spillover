# K050: structured sparsity and matched runtime investigation

**At nonzero κ, seven-site checkpoints obtain larger native-relative speedups primarily because
their native baselines take longer. They do not have lower absolute K050 latency
than the matched four-site checkpoints.** The counters also reject the proposed
explanation of more exploitable projection sparsity for seven-site OL1: four-site
OL1 bypasses more projection MMAs and benefits more from projection skipping at
every matched threshold.

This analysis uses the existing 30-checkpoint cohort, c01–c30. No training,
evaluation, benchmarking, or manuscript modification was performed.

## The decisive pair

At κ = 0.5, comparing c20 and c30:

| Quantity | 4-site + OL1 | 7-site + OL1 |
|---|---:|---:|
| Model-wide sparsity (%) | 12.713 | 27.483 |
| Projection contribution (pp of model work) | 12.657 | 10.709 |
| QK + PV contribution (pp of model work) | 0.057 | 16.773 |
| Projection MMAs bypassed (%) | 82.30 | 57.68 |
| Projection arithmetic-avoidance proxy (%) | 75.75 | 41.88 |
| Projection SIMT products | 116,305,664 | 241,605,760 |
| QK MMAs skipped (%) | 0.00 | 58.00 |
| PV MMAs skipped (%) | 10.42 | 67.21 |
| Native geometric-mean latency (ms) | 0.73944 | 0.84409 |
| K050 all-skips-off latency (ms) | 0.62263 | 0.62071 |
| K050 attention-dense latency (ms) | 0.45175 | 0.46837 |
| Full K050 latency (ms) | 0.45948 | 0.47337 |
| Fusion-only control speedup, common native reference | 1.188× | 1.360× |
| Projection-sparse gain | 1.378× | 1.325× |
| Attention-sparse gain | 0.9832× | 0.9894× |
| Full native-relative speedup | 1.609× | 1.783× |

The speedup ratio is exactly `(native7/native4) / (K0507/K0504)`: a 14.15%
slower native baseline divided by a 3.02% slower optimized implementation yields
10.80% greater relative speedup. The native implementation separately applies
Q/K/V absolute-value, comparison and masking operations for nonzero κ; at κ = 0
those gates return their inputs. K050 fuses those gates with QKV layout/RoPE,
including in the all-skips-off control. The similar all-skips-off latencies and
κ = 0 behavior support native gate overhead as an explanation. They do not
provide an isolated per-operation timing attribution.

## What the kernel can exploit

QKV and FFN-up bypass a 16×8×16 MMA when its entire 16-token × 16-feature
activation fragment is zero. FFN-down and attention output instead combine
scalar execution for safe rows with at most two nonzeros with MMA skipping over
the remaining 8-token × 16-feature support. Their MMA bypass counters therefore
include **replacement by scalar work**, not just eliminated work.

The high-threshold projection bypass fractions are:

| Projection | 4-site + OL1 | 7-site + OL1 |
|---|---:|---:|
| QKV | 67.12% | 8.44% |
| FFN-up | 54.49% | 0.00% |
| FFN-down | 97.47% | 94.50% |
| Attention output | 99.98% | 99.64% |

Thus seven-site does not hide superior projection structure behind its smaller
scalar contribution in this pair. Its smaller opposing attention overhead
partially offsets its weaker projection benefit, but its absolute K050 latency
remains higher.

Across all ten matched pairs, four-site has greater projection-sparse gain.
Across all five OL1 pairs it also has higher projection scalar contribution and
MMA bypass. Without pressure, the small bypass differences are not uniform:
seven-site bypasses slightly more at κ = 0 and 0.1. Across all eight nonzero-κ
pairs, seven-site nevertheless reports higher relative speedup and higher
absolute K050 latency. Differences near κ = 0 are tiny; no significance claim
is made. [All matched values and differences](data/matched-pairs.csv) are retained.

Attention skips an MMA when **either complete operand fragment** is zero.
For 29 checkpoints, QK skips 0% and PV skips 10.42%; the latter includes masked
or padded probability work. Only c30 has additional skips. Attention skipping
still increases latency for every checkpoint, with gains between 0.9832× and
0.9894×. Four-site incurs the larger relative attention overhead in nine of ten
pairs; the exception is pressure-free κ = 0.05. These small differences are
descriptive, not significance tests.

## Associations

Each observation is one checkpoint, weighted equally. OLS includes an intercept.

| Predictor → runtime quantity | Pearson r | OLS R² |
|---|---:|---:|
| S_model → full speedup | 0.904 | 0.817 |
| Projection scalar zero fraction → projection gain | 0.643 | 0.413 |
| Projection MMA bypass fraction → projection gain | 0.973 | 0.946 |
| Attention scalar zero fraction → attention gain | 0.355 | 0.126 |
| Attention MMA skip fraction → attention gain | 0.373 | 0.139 |

The attention-MMA fit has only **two distinct predictor values, with c30 alone
at the higher value**. It is effectively a single-checkpoint contrast. Within
baseline/local and four-site groups the predictor is constant and correlation
is undefined, not zero.

The S_model/speedup R² values within baseline/local, four-site and seven-site
families are 0.542, 0.784 and 0.762. Family means alone explain R² = 0.499;
centering both variables within family gives R² = 0.660. The global relationship
therefore is not solely a difference between family means, but these descriptive
fits do not isolate sparsity from recipe, threshold, pressure weight, quality,
or native execution cost. [Complete stratified fits](data/associations.csv)
include raw-latency and native-normalization sensitivity checks.

## Evidence and limits

**Directly recorded:** canonical logical counters; instrumented FFN-down/output
and attention MMA counters, SIMT products and row histograms; all raw timing
pairs. **Reconstructed:** QKV/FFN-up skip counts from actual BF16 operands by the
original diagnostic; pooled fractions, latency geometric means, ratios and fits
here. **Unavailable:** pure-zero-only versus SIMT-replacement bypass counts,
unmasked-only attention MMA counts, skip counters restricted to the timed input
subset, and per-operation timing attribution for all 30 checkpoints.

Logical sparsity is the retained canonical FP16 measurement. Kernel diagnostics
use actual BF16 operands over all 338 validation blocks; timings use the original
64-block subset, seven paired passes and three fresh processes per implementation
on RTX5090. These are aligned checkpoints but different precision/coverage
estimands. The exported BF16 scalar lower bounds do not include probability
underflow and are not replacement S_model measurements.

The [methods and field guide](METHODS.md) traces the code, defines denominators
and lists the minimal instrumentation needed to fill each gap. The
[30-row CSV](data/checkpoints.csv) contains all 140 fields; blanks explicitly
denote unavailable quantities. The [five-page diagnostic PDF](figures/kernel-investigation.pdf)
covers associations, within-family fits, both matched sweeps and absolute
latencies. Source hashes are recorded in [provenance.json](data/provenance.json).

**Conclusion:** the larger seven-site speedup is relative to a slower native
reference. Four-site OL1 actually exposes more projection skipping and has lower
full K050 latency at matched κ. Seven-site's additional attention skips
do not yield a net attention-path benefit in this cohort.

## Reproduce

Run from the repository root, using retained files only:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/investigation/01_extract.py
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/investigation/02_analyze.py
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/investigation/03_plot.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/investigation/test_investigation.py -q
```
