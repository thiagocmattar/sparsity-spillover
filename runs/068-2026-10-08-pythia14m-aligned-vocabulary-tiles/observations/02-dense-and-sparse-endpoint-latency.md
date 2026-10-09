# Dense, no-skip and sparse vocabulary-head execution

Question: does the aligned input gate let the custom head beat native dense execution?

Method and coverage: Both models are fresh random-initialization training, seed1234,712 updates. Only head κ varies (.1/.5); h/z8×16 κ0 and seven-site Ph+head pressure are matched. All500 validation documents yield338 complete2048-token blocks,692224 input tokens and691886 predictions;1444 tail tokens are excluded. Native training evaluation uses B32; the endpoint uses B1 on one RTX5090. The head gate is sparse activation masking; training and the native head baseline use dense matmul. All h/z projections remain native dense in the endpoint comparison. Only custom sparse head execution bypasses MMA.

Each model has three fresh qualified processes. Each boundary uses64 fixed validation inputs×7 randomized paired passes,
with all three modes measured on each pair. The primary timer is synchronized host wall time; CUDA-event measurements
use separate replays. Geometric means first reduce each process, then the three processes. Error bars span the three
process geometric means and are descriptive variability, not confidence intervals over training seeds.

Sources: [endpoint reduction](../inference/results/comparison.json), [paired raw timing rows](../inference/artifacts/attempts/),
and [byte identities](../results/figure-sources.json).

Caption: three separate boundaries: projection only; head gate+cast+projection; complete model forward.
Blue=Native dense, orange/dots=Custom no-skip, green/hatching=Custom sparse, with identical colors across both models
and panels. Panels have individually labeled millisecond scales. The boundaries overlap and cannot be summed or subtracted
to form a measured decomposition. No-skip isolates bypass within the custom package; the dense comparison also includes
layout/preparation differences. Static weight packing, input staging, loss, softmax and token selection are outside timing.
All50304 logits are computed and written.

Result: the full table and process ratios are in [REPORT.md](../inference/results/REPORT.md). The figure shows absolute
latencies so sparsity and bypass can be assessed against the costs that remain.

Caveats: full-sequence B1/T2048 inference, one RTX5090, one model size and seed. These timings do not establish
cached-decoding performance or HBM traffic. All transformer h/z projection kernels stay native dense.
The two trained models have different weights; comparing their absolute latency is not a pure runtime sparsity ablation.
Within each model, all three modes use the same weights, gate and vocabulary contract.
This observation is not a promoted manuscript claim.

Source script: [19_figures.py](../19_figures.py). Output: [PDF](../figures/02-dense-and-sparse-endpoint-latency.pdf).
