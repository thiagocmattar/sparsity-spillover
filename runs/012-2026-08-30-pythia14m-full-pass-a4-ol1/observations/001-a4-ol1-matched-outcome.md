# A4-OL1 matched outcome

## Question

Does operational OL1 pressure at all active A4-Z sites change the
quality--logical-opportunity frontier relative to threshold-only A4-Z at the
same gate threshold?

## Method

Run 012 applied one-sided A4-Z gates at `{a,m,h,z}` and orthogonal L1
pressure at those same sites with `lambda=1` and step budget 1. Each row is
matched to Run 011 at the same `kappa`. The runs share the Pythia-14M random
initialization, MiniPile cache and order, optimizer schedule, 712-boundary
budget, validation coverage, and diagnostic definitions. The result table is a
direct rendering of Run 012's verified cohort artifact; Run 011's verified
artifact supplies the comparator columns.

## Coverage

- Five Run 012 conditions completed 712 optimizer boundaries each: 3,560
  boundaries and 7,465,861,120 training input tokens in total.
- The verifier accepted 20 complete validation passes. Every pass covered all
  338 complete 2,048-token blocks (692,224 input tokens) from all 500
  validation documents and excluded the declared 1,444-token tail.
- All five initial-parameter and schedule hashes match. Every final checkpoint,
  activation/weight diagnostic, logical-product count, OL1 boundary record, and
  transfer inventory passed standalone and cohort verification.

## Result

| `kappa` | A4-OL1 validation loss | Matched A4 loss | loss delta (OL1 - A4) | A4-OL1 `R_model` | Matched A4 `R_model` | `R_model` delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.215749 | 5.470497 | -0.254749 | 8.4094% | 7.2120% | +1.1974 pp |
| 0.01 | 5.204504 | 5.466500 | -0.261996 | 8.5858% | 7.4137% | +1.1721 pp |
| 0.05 | 5.195590 | 5.434110 | -0.238520 | 9.0356% | 8.2059% | +0.8297 pp |
| 0.1 | 5.228687 | 5.419642 | -0.190955 | 9.3435% | 8.9530% | +0.3905 pp |
| 0.5 | 5.722666 | 5.659680 | +0.062986 | 10.2274% | 10.2155% | +0.0119 pp |

At matched `kappa <= 0.1`, A4-OL1 has both lower validation loss and higher
logical-product opportunity than threshold-only A4-Z. The gain shrinks as the
threshold rises. At `kappa=0.5`, OL1 adds essentially no matched
`R_model` and increases validation loss, reversing the lower-threshold
quality improvement.

Within Run 012, `kappa=0.05` has the lowest validation loss (5.195590) at
`R_model=9.0356%`. Moving to `kappa=0.1` adds 0.3079 percentage points of
`R_model` for 0.033097 higher loss. The `kappa=0.5` condition reaches the
largest measured `R_model` (10.2274%) but with a much larger loss increase.

## Legend/caption

No figure is attached. The table reports complete-validation cross-entropy and
logical-product opportunity. Negative loss delta favors A4-OL1; positive
`R_model` delta means more logical multiplication opportunities. Neither
`R_model` nor `R_block` is a measured runtime speedup.

## Caveats

This is one seed, one model scale, and one MiniPile pass. Because A4 gates and
OL1 pressure act jointly at the same four sites, the result does not identify
site-level causal contributions. The Transformers/PyTorch recipe mapping is
not GPT-NeoX-bitwise. At `kappa=0.5`, the final selected-site exact-zero
fractions exceed 99.9% at both `h` and `z`; reduced pressure-gradient
support is a plausible interpretation of the reversal, not a demonstrated
mechanism. No runtime kernel speedup was measured.

## Sources

- Scientific result and matched comparison:
  [`artifacts/verification.json`](../artifacts/verification.json)
- Run 012 implementation and design: [`README.md`](../README.md)
- Run 011 matched comparator:
  [`artifacts/verification.json`](../../011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json)
- Generating verifier: [`03_verify.py`](../03_verify.py) and
  [`verification.py`](../verification.py)
