# Runtime Summary

Run 029, RTX5090, BF16 batch 1, T=2048, native SDPA CUDA-graph denominator, 3 fresh processes. Geomeans for P0 cover only its five qualified checkpoints; they are not cohort-matched rankings. Source O007.

| Candidate | Qualified checkpoints | Speedup geomean | Minimum | Maximum | Above 1x |
| --- | --- | --- | --- | --- | --- |
| k049 | 35 | 1.123355 | 0.954967 | 1.528395 | 29 |
| k050 | 35 | 1.250205 | 1.005938 | 1.783175 | 35 |
| k050-attention-dense | 35 | 1.267131 | 1.018479 | 1.802220 | 35 |
| k050-no-skip | 35 | 1.183457 | 1.050314 | 1.362831 | 35 |
| p0 | 5 | 0.867570 | 0.779934 | 1.001212 | 1 |
