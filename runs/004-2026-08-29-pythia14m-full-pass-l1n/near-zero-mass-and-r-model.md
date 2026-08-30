# Run 004 near-zero mass and R_model

All entries are percentages from the reloaded final checkpoints. Near-zero mass
uses `|x| <= 1e-3` and is pooled count-first over all six layers and all 338
complete validation blocks. The validation coverage is 500 MiniPile documents,
692,224 evaluated input tokens, and a reported but excluded 1,444-token tail.

`R_model` uses exact-zero logical products, not the `1e-3` threshold. It is the
measured block zero-product count divided by the block plus dense LM-head model
product count. It is a logical opportunity, not removed FLOPs or measured
speedup.

| Activation | Condition | L1N lambda | `h` near-zero (%) | `m` near-zero (%) | `q_post` near-zero (%) | `k_post` near-zero (%) | `v` near-zero (%) | Attention output near-zero (%) | `R_model` (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeLU | control | n/a | 0.237736 | 0.070630 | 0.065518 | 0.059550 | 0.129597 | 0.701354 | 0.000001142 |
| ReLU | control | n/a | 63.508542 | 0.067001 | 0.050349 | 0.045815 | 0.132739 | 0.480568 | 2.714129933 |
| ReLU | L1N | 0.05 | 73.501792 | 0.070568 | 0.056967 | 0.045499 | 0.149564 | 0.697766 | 3.141594862 |
| ReLU | L1N | 0.1 | 78.047723 | 0.067589 | 0.055263 | 0.055462 | 0.155159 | 0.801612 | 3.336306480 |
| ReLU | L1N | 0.5 | 88.579031 | 0.071537 | 0.057830 | 0.049814 | 0.180704 | 0.855134 | 3.787578106 |
| ReLU | L1N | 1.0 | 92.351206 | 0.063650 | 0.051757 | 0.050614 | 0.185325 | 0.724827 | 3.949334259 |

Run 004 did not evaluate L1N pressure for GeLU, so its only row is the GeLU
control. `Attention output` is the output of `attention.dense` (`W_o`) before
residual addition; it is not the pre-`W_o` context site `z`.

## Provenance

- Condition and attempt mapping: `artifacts/verification.json`
- Near-zero integer counts: `artifacts/attempts/*/diagnostics/activation_statistics.json`
- Exact-zero logical-product counts: `artifacts/attempts/*/diagnostics/logical_products.json`
- Count-first near-zero reductions: `artifacts/figure_data_sitewise.json` and `artifacts/figure_data_attention_output.json`

