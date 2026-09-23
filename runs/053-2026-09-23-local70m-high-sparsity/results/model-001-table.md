# Local full-model integration

One local process per checkpoint; full-validation numerical qualification, descriptive timings; the kappa .1 candidate fails numerical qualification.

| Checkpoint | Native | opt073 | Best tested dense | Prior policy | New candidate | Skip off | Candidate / native Base | Candidate qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| c00 | 6.0552 | 8.5301 | 5.2099 | 5.2195 | 5.1114 | 5.1116 | 0.8441 | True |
| c26 | 6.2727 | 4.6729 | 5.0728 | 4.6000 | 4.4959 | 5.3430 | 0.7425 | True |
| c24 | 6.2190 | 7.1437 | 5.0282 | 4.6561 | 4.6271 | 5.3138 | 0.7642 | True |
| c25 | 5.7522 | 6.1763 | 4.6475 | 4.2910 | 4.2282 | 4.8849 | 0.6983 | False |

Geometric-mean synchronized host milliseconds. c00=Base; c24/c25/c26=T2/Ph kappa .05/.1/.5.
The new candidate fails at kappa .1 on input78. Its timing is retained but is not a qualified speedup. No statistical or cross-size claim follows.
Source: `20_reduce_model_with_failure.py`; summary JSON contains source hashes, loss differences, diagnostics and profile summaries.
