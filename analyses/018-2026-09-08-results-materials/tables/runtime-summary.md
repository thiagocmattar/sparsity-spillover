# Runtime Summary

Run 029 restricted to the 30 included checkpoints; RTX5090, BF16 batch 1, T=2048, native SDPA CUDA-graph denominator, 3 fresh processes. P0 covers only four qualified checkpoints; it is not a cohort-matched ranking. Source O007.

| Candidate | Qualified checkpoints | Speedup geomean | Minimum | Maximum | Above 1x |
| --- | --- | --- | --- | --- | --- |
| k049 | 30 | 1.114055 | 0.954967 | 1.528395 | 24 |
| k050 | 30 | 1.233992 | 1.005938 | 1.783175 | 30 |
| k050-attention-dense | 30 | 1.250559 | 1.018479 | 1.802220 | 30 |
| k050-no-skip | 30 | 1.182858 | 1.050314 | 1.362831 | 30 |
| p0 | 4 | 0.890976 | 0.817961 | 1.001212 | 1 |
