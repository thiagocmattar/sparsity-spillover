# Findings for the introduction paragraph

10 September 2026. These notes address the comments immediately before
“We pretrain Pythia-family…” in `introduction.tex`. They summarize existing
results for the proposed paragraph; they do not change the TeX or promote new
entries into the research finding registry. The kernel paragraph is outside
this edit's scope.

## Study context

Pythia supplies a common architecture family across parameter scales, allowing
controlled comparisons of the same intervention recipes when pretraining from
random initialization. The primary study contains **30 conditions at 14M** and
**12 each at 70M and 410M**, for 54 total. The five historical h-only-pressure
A4 conditions are separate and should not inflate the primary 14M count.
The larger cohorts test selected recipes rather than repeating all 14M ablations.
This is one-seed evidence under the stated training budget; the 410M learning
rate differs. [Endpoint and protocol evidence](../../analyses/018-2026-09-08-results-materials/figure_data.json)
and [scale comparison](../../analyses/018-2026-09-08-results-materials/observations/O003-scale-transfer.md).

## Findings

1. **Small activation magnitudes and exact zeros are different outcomes.**
   L1 pressure encourages smaller magnitudes, while thresholding explicitly
   maps a region of values to zero. In the 14M endpoint distributions,
   A4-OL1 at kappa = 0.5 has a sharp nonzero attention peak around zero but
   only **0.22% attention zeros**; A7-OL1, which directly thresholds attention,
   has **95.60%**. At kappa = 0, A7-OL1 applies pressure to attention but its
   symmetric gate is the identity, and its attention zero fraction is **below
   0.01%**. These observations support distinguishing concentration near zero
   from exact sparsity. They do not establish that L1 “does not converge to
   zero”: the records are endpoints, and the local FFN pressure conditions
   already use ReLU. The distribution comparison also changes the pressure
   targets, so it does not isolate a threshold-only causal effect.
   [Distribution observation and zero-mass table](../../analyses/018-2026-09-08-results-materials/observations/O011-activation-density-v3.md).

2. **Broader thresholding produces substantially more model-wide sparsity
   than local pressure; their combination produces the largest observed
   sparsity.** At 14M, local ReLU plus naive L1 reaches **3.95%** against a
   **4.28% local-site ceiling**, compared
   with **15.39%** for seven-site thresholding at kappa = 0.5. Adding OL1
   to that same seven-site threshold configuration raises sparsity to
   **27.48%**, against its **29.95% ceiling**, for **+0.1265 validation loss**
   (5.7029 to 5.8294). This matched pressure addition is **+12.096 percentage
   points**. The local-versus-broader comparison also changes affected sites
   and therefore available computation; it is not an isolated ranking of
   pressure against thresholding. Pressure's marginal value is conditional:
   at the same threshold, its four-site addition gives **+2.498 pp** for
   **+0.3783 loss**. [Paired effects](../../analyses/018-2026-09-08-results-materials/observations/O002-blocked-effects.md)
   and [unrounded endpoints](../../analyses/018-2026-09-08-results-materials/figure_data.json).

3. **FFN and attention activations respond differently, with no universal
   ordering of their sparsity.** Under A7-OL1 at 14M, the pooled exact-zero
   fractions are:

   | Trained threshold | FFN h,m | Attention q,k,v |
   | --- | ---: | ---: |
   | 0 | 64.52% | <0.01% |
   | 0.05 | 75.29% | 7.26% |
   | 0.5 | 93.64% | 95.60% |

   Thus “FFNs are sparser at moderate thresholds” is supported; “FFNs always
   sparsify more easily than attention under the same intervention” is too
   strong. At the largest threshold attention slightly exceeds FFNs. Also,
   the maps differ: FFN gates are one-sided, while q/k/v gates are symmetric;
   the same numerical threshold is not the same zeroing rule or a matched
   activation-scale cutoff. These are element-pooled group rates, not claims
   about every layer/site. [Exact masses and pooling](../../analyses/018-2026-09-08-results-materials/activation-density-v3-data.json).

4. **The aggressive seven-site recipe extends the favorable high-sparsity
   region beyond uniform clipping of the GELU baseline at every size.**
   For A7-OL1 at kappa = 0.5:

   | Size | Model-wide sparsity | Seven-site ceiling | Validation loss | A0 clipping at p = 0.6: sparsity / loss |
   | --- | ---: | ---: | ---: | ---: |
   | 14M | 27.48% | 29.95% | 5.8294 | 7.76% / 6.8212 |
   | 70M | 40.60% | 49.42% | 5.2159 | 22.41% / 5.6578 |
   | 410M | 80.62% | 87.25% | 5.1207 | 44.10% / 6.7213 |

   This trained recipe has both lower loss and higher sparsity than every
   tested A0 clipping point at p = 0.6, 0.7, 0.8 and 0.9, at all three sizes.
   This is a regional comparison, not dominance at every quality budget.
   Seven-site placement also has broader reach than the four-site clipping
   control, whose ceilings are 12.83%, 37.06% and 74.78%. Thus the difference
   combines learned representations, transformations and placement with
   intervention timing. [Scale evidence](../../analyses/018-2026-09-08-results-materials/observations/O003-scale-transfer.md)
   and [retained clipping comparison](../../analyses/020-2026-09-09-intervention-revision/data/cross-size-audit.json).

5. **ReLU pretraining improves subsequent clipping in an intermediate region,
   and its advantage grows across the tested sizes at a fixed target.**
   At calibration quantile p = 0.5, replacing GELU pretraining with ReLU
   pretraining gives both lower loss and higher achieved sparsity after clipping:

   | Size | GELU + clipping: loss / sparsity | ReLU + clipping: loss / sparsity | Loss reduction | Sparsity gain |
   | --- | ---: | ---: | ---: | ---: |
   | 14M | 6.0777 / 6.44% | 5.9763 / 6.99% | 0.1015 | 0.553 pp |
   | 70M | 4.8331 / 18.59% | 4.5221 / 22.52% | 0.3110 | 3.927 pp |
   | 410M | 5.5855 / 36.80% | 4.9757 / 45.62% | 0.6098 | 8.815 pp |

   Both advantages increase across these three sizes **at this target**.
   The favorable same-target region is p = 0.4–0.6 at 14M, 0.4–0.8 at 70M,
   and 0.4–0.7 at 410M. Its extent therefore does not increase monotonically.
   ReLU has an initial loss cost of +0.0611, +0.1230 and +0.1038; low-target
   total loss still favors GELU, and extreme targets reverse the loss ordering
   at 14M/410M. Incremental clipping damage relative to each checkpoint's own
   measured p = 0 is smaller for ReLU at every positive tested target, which
   is distinct from having lower total loss. A matched quantile is not matched
   achieved sparsity, and architecture affects the model-wide percentages.
   [Identity-checked comparison and all targets](../../analyses/020-2026-09-09-intervention-revision/data/cross-size-audit.json)
   and [source observation](../../analyses/020-2026-09-09-intervention-revision/observations/O003-cross-size.md).

## Wording priorities

Use the paragraph to connect controlled pretraining, 14M ablations, the paired
threshold/pressure gain, differing FFN/attention responses, and the larger-size
recipe/clipping results. Pair headline sparsity with its ceiling and quality
cost. Express the ReLU size trend through the named p = 0.5 comparison, rather
than claiming that an entire frontier universally improves with model size.

Verification used the retained unrounded endpoint, histogram and clipping JSON
records. The three input hashes recorded by the cross-size audit match their
current source files. The proposed values were recomputed from those records;
no training, evaluation, figure regeneration or TeX edit was performed.
