# 70M final kernel: retained checkpoints

S_model is canonical pooled FP16 logical-product sparsity. Native/candidate latency is BF16, B1/T2048, full50304 logits. Each point pools1344 timing pairs in three fresh processes. Process ranges are not confidence intervals. Failed qualification is retained and cannot support a runtime gain.

| Family | Kappa | S_model (%) | Native (ms) | Port (ms) | Speedup | Qualified |
|---|---:|---:|---:|---:|---:|---|
| A0 | N/A | 0.000454 | 1.663883 | 3.313247 | 0.5022x | True |
| A1-H | N/A | 10.065758 | 1.658539 | 3.318765 | 0.4997x | True |
| A4+OL1@all | 0 | 25.672509 | 1.786767 | 2.798659 | 0.6384x | True |
| A4+OL1@all | 0.01 | 26.224248 | 1.782935 | 2.712373 | 0.6573x | True |
| A4+OL1@all | 0.05 | 28.489958 | 1.782524 | 2.318007 | 0.7690x | True |
| A4+OL1@all | 0.1 | 30.574258 | 1.785444 | 2.074882 | 0.8605x | True |
| A4+OL1@all | 0.5 | 35.596219 | 1.779589 | 1.626444 | 1.0942x | True |
| A7+OL1@all | 0 | 23.562417 | 1.786377 | 3.185773 | 0.5607x | True |
| A7+OL1@all | 0.01 | 24.226276 | 1.922832 | 3.139875 | 0.6124x | True |
| A7+OL1@all | 0.05 | 26.523941 | 1.920101 | 2.997303 | 0.6406x | True |
| A7+OL1@all | 0.1 | 28.693785 | 1.920130 | 2.890846 | 0.6642x | True |
| A7+OL1@all | 0.5 | 40.601872 | 1.922504 | 1.748420 | 1.0996x | True |
| A4+OL1@h | 0 | 25.791632 | 1.783114 | 2.627091 | 0.6787x | True |
| A4+OL1@h | 0.01 | 25.970787 | 1.782908 | 2.586465 | 0.6893x | True |
| A4+OL1@h | 0.05 | 26.566402 | 1.785198 | 2.258318 | 0.7905x | True |
| A4+OL1@h | 0.1 | 27.218067 | 1.785462 | 1.987783 | 0.8982x | True |
| A4+OL1@h | 0.5 | 29.235859 | 1.782363 | 1.619968 | 1.1002x | True |
| A7+OL1@h | 0 | 25.786062 | 1.783113 | 2.657395 | 0.6710x | True |
| A7+OL1@h | 0.01 | 26.008979 | 1.920143 | 2.569484 | 0.7473x | True |
| A7+OL1@h | 0.05 | 26.894689 | 1.921498 | 2.246481 | 0.8553x | True |
| A7+OL1@h | 0.1 | 27.779141 | 1.921822 | 1.972721 | 0.9742x | True |
| A7+OL1@h | 0.5 | 32.232517 | 1.922646 | 1.618370 | 1.1880x | True |
