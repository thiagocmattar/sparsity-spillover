# Approved vocabulary and notation

The author approved **architectural reach** and **R_arch** on 9 September 2026.
US academic English is used. Experimental recipe IDs remain unchanged.

| Expression | Meaning and boundary |
|---|---|
| Activation pressure / naive L1 / orthogonal L1 (OL1) | Training objective and update choices; smaller magnitudes do not guarantee exact zeros or preserved task loss. |
| Thresholding nonlinearity | The stated one-sided or symmetric map; equality survives. One-sided maps remove all negative values. |
| Thresholding sites N / pressure sites P | Independent design fields; expanding the equal-target mean also changes existing targets' weights. |
| Activation sparsity at site s | Exact-zero element fraction at that site; identify h/m or q/k/v pooling explicitly. |
| Model-wide activation sparsity S_model | Count-pooled zero-operand scalar multiplications divided by the declared full-model multiplication workload. |
| Architectural reach R_arch(N) | Former S_model^max: the unchanged all-zero selected-site reachable-product numerator divided by the unchanged model denominator. Architecture, selected sites and sequence length determine it. Natural zeros outside reach can contribute to observed S_model. |
| Block-only sparsity S_block | Same measured numerator divided by block products; equals the existing common-A7 U_arch normalization in this declared graph. |
| Same-size A0 reference | Quality reference; distinct from native execution of a sparse checkpoint. |
| Native execution | Native PyTorch/SDPA execution of the same checkpoint, BF16, batch one, uncached T=2048 with full logits, RTX5090. |
| Matched fused control, all sparse paths disabled | K050 no-skip ablation; separate from native and from attention-dense. |
| Uniform post-hoc magnitude clipping | Four-site, separately calibrated site/layer quantiles; not optimized TEAL allocation or a matched final-transformation adaptation test. |

## Alias migration

`S_model^max` / TeX `\Smax` becomes `R_arch` / `\Rarch`. The numerator,
denominator, propagation rules and values remain identical. `U_arch` remains a
raw data identifier; the cross-size plot labels its common-A7 normalization as
block-only sparsity. Operational `R_model`, `R_block`, `R_model_max*` and all
recipe identifiers remain unchanged. The final source, captions, PDF labels,
tables and supplement explanations must be checked together; archived records
retain their historical spelling.

## Numerical conventions

Compute with full-precision records and integer counts; round only for display.
Loss differences are treatment minus the named reference. Sparsity differences
are percentage points. Timing ranges are checkpoint ranges, not confidence
intervals. Distinguish canonical FP16 counts/loss, BF16 timed execution and
BF16 diagnostic lower bounds. Retrospective loss budgets and sensitivity checks
are derived analyses, not newly executed or preregistered experiments.
