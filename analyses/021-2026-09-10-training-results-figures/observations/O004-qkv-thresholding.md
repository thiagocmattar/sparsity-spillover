# O004 - Adding Q/K/V thresholding without pressure

## Question and status

At a matched trained threshold, what changes when Q/K/V thresholding is added
to the four-site recipe without optimization pressure? The author requested
this manuscript table and analysis on 11 September 2026. This summarizes
retained endpoints; no new training or checkpoint evaluation is performed.

## Method, sources and coverage

Ten Pythia-14M endpoints form five A7-minus-A4 pairs at kappa = 0, 0.01, 0.05,
0.1 and 0.5. A4 (Run 011) thresholds a,m,h,z; A7 (Run 013) additionally applies
symmetric thresholds to post-RoPE q,k and to v. Both use no pressure.

The [Analysis 018 evidence](../../018-2026-09-08-results-materials/figure_data.json)
contains the trained endpoints and the five `A4 to A7 gates` contrasts.
Source SHA-256: `d1dfd326bde98ab88a827f3f72c46ded9c904128c0cc803128c322f6de398a05`.
The original reduction is
[`01_build.py`](../../018-2026-09-08-results-materials/01_build.py), using
[`evidence.py`](../../018-2026-09-08-results-materials/evidence.py);
the earlier interpretation is
[O002](../../018-2026-09-08-results-materials/observations/O002-blocked-effects.md).

All five pairs match initial-parameter hash, data-order schedule, model/data
seeds (1234), optimizer settings, 712 training steps, 1,493,172,224 training
input tokens and validation-cache identity. Loss and sparsity are from the
same eager full-validation pass over all 500 MiniPile validation documents:
338 complete 2,048-token sequences, 692,224 input tokens and 691,886 next-token
predictions, excluding the 1,444-token tail. These are not terminal-log losses.

Sparsity is 100 times the pooled block zero-product count divided by
6,363,055,915,008 model products. The denominator includes the final dense
output projection and excludes future-masked attention pairs. Differences
are computed before rounding; percentage points are not relative percentages.

## Table and results

The [manuscript table](../../../manuscript/draft/tables/qkv-thresholding-no-pressure.tex)
reports raw endpoints, with the highest-threshold row in bold. Loss is rounded
to four decimals and model-wide sparsity to three decimals in percent.

| Kappa | A4 loss | A7 loss | A4 sparsity (%) | A7 sparsity (%) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.4705 | 5.4684 | 7.212 | 7.218 |
| 0.01 | 5.4665 | 5.4588 | 7.414 | 7.618 |
| 0.05 | 5.4341 | 5.4379 | 8.206 | 9.127 |
| 0.1 | 5.4197 | 5.4287 | 8.953 | 10.425 |
| **0.5** | **5.6597** | **5.7029** | **10.216** | **15.387** |

At kappa = 0.5, A7 minus A4 is +5.1712760024 pp model-wide sparsity and
+0.0432166468 validation loss. The manuscript states +5.17 pp / +0.043 loss
in prose, without adding difference columns to the table. The sparsity
difference grows across the five thresholds; the loss difference changes sign.

Caption: **Broader threshold placement increases model-wide sparsity at
matched threshold without pressure.** A4 thresholds a,m,h,z, whereas A7
additionally thresholds q,k,v. At kappa = 0, the additional Q/K/V nonlinearities
are identities, and the two recipes are nearly identical. As kappa increases,
their sparsity difference grows; at kappa = 0.5, A7 reaches 15.39% model-wide
sparsity versus 10.22% for A4, with validation losses 5.7029 and 5.6597,
respectively.

## Interpretation limits

- One matched seed; no seed-robustness or statistical-significance inference.
- At kappa = 0, the added symmetric gates are identities. The small endpoint
  residual is not evidence of a nontrivial threshold intervention there.
- These are separately trained recipes. The comparison measures the full
  training effect of adding Q/K/V thresholds, not post-hoc clipping or an
  isolated inference-time change to a fixed checkpoint.
- Model-wide sparsity counts logical zero-operand opportunities, not speedup.
- This pressure-free comparison does not change the normalization of a
  pressure objective. It is distinct from the OL1 target-set caveat in O003.

## Manuscript adoption and verification

Adopted in Section 4.3, with Table 2 on page 7. All 25 table cells were checked
against the source, including the five matched identity records and pooled
integer-count ratios. The five differences agree with the retained contrasts.
Two LaTeX passes produced a 27-page draft with resolved references and no
overfull boxes; three existing underfull vertical-box warnings remain.
Pages 6-8 were rendered with Poppler and visually checked. All PDF text lies
within page bounds, fonts are embedded, provenance links resolve, and all
three frozen Analysis 021 figure PDFs are unchanged.
