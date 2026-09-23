# Wider sparse component screen

Numerically qualified and >=5% faster than best qualified dense/prior control at both moderate kappas in both prefixes. Component shortlist only.

| Site | Best candidate | Worst moderate ratio | Shortlisted |
|---|---|---:|---|
| h.0 | tiles_m32 | 1.6366 | False |
| h.1 | union_m32 | 2.0278 | False |
| h.2 | union_m32 | 2.0614 | False |
| h.3 | union_m32 | 2.2722 | False |
| h.4 | union_m32 | 2.1224 | False |
| h.5 | union_m32 | 2.2879 | False |
| z.0 | tiles_m32 | 1.4065 | False |
| z.1 | union_m32 | 0.9750 | False |
| z.2 | union_m32 | 1.1054 | False |
| z.3 | union_m32 | 1.3720 | False |
| z.4 | tiles_m32 | 1.3216 | False |
| z.5 | tiles_m32 | 1.3111 | False |

Ratios below1 are faster. These are complete component paths on local training inputs, not full-model or RTX5090 results.
Sources: `09_reduce_components.py` and the source-hashed summary JSON.
