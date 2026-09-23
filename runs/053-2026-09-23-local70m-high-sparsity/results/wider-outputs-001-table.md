# Wider sparse component screen

Numerically qualified and >=5% faster than best qualified dense/prior control at both moderate kappas in both prefixes. Component shortlist only.

| Site | Best candidate | Worst moderate ratio | Shortlisted |
|---|---|---:|---|
| h.0 | union_m32_n256 | 1.2230 | False |
| h.1 | union_m32_n256 | 1.1253 | False |
| h.2 | union_m32_n256 | 0.9238 | True |
| h.3 | union_m32_n256 | 0.9105 | True |
| h.4 | union_m32_n256 | 0.9187 | True |
| h.5 | union_m32_n256 | 0.9913 | False |
| z.0 | union_m32_n256 | 0.9862 | False |
| z.1 | union_m32_n256 | 0.6180 | True |
| z.2 | union_m32_n256 | 0.6808 | True |
| z.3 | union_m32_n256 | 0.8296 | True |
| z.4 | union_m32_n256 | 0.8705 | True |
| z.5 | union_m32_n256 | 0.8040 | True |

Ratios below1 are faster. These are complete component paths on local training inputs, not full-model or RTX5090 results.
Sources: `13_reduce_wider_outputs.py` and the source-hashed summary JSON.
