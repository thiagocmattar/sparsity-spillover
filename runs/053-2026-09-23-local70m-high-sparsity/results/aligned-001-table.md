# Wider sparse component screen

Numerically qualified and >=5% faster than best qualified dense/prior control at both moderate kappas in both prefixes. Component shortlist only.

| Site | Best candidate | Worst moderate ratio | Shortlisted |
|---|---|---:|---|
| h.0 | tiles_m32_n128 | 1.1455 | False |
| h.1 | tiles_m32_n128 | 3.1721 | False |
| h.2 | tiles_m32_n128 | 2.5809 | False |
| h.3 | tiles_m32_n128 | 2.7110 | False |
| h.4 | tiles_m32_n128 | 2.4554 | False |
| h.5 | tiles_m32_n128 | 2.2751 | False |
| z.0 | tiles_m32_n128 | 1.1198 | False |
| z.1 | tiles_m32_n128 | 0.9629 | False |
| z.2 | tiles_m32_n128 | 1.0129 | False |
| z.3 | tiles_m32_n128 | 1.0859 | False |
| z.4 | tiles_m32_n128 | 1.0271 | False |
| z.5 | tiles_m32_n128 | 1.0541 | False |

Ratios below1 are faster. These are complete component paths on local training inputs, not full-model or RTX5090 results.
Sources: `24_reduce_aligned.py` and the source-hashed summary JSON.
