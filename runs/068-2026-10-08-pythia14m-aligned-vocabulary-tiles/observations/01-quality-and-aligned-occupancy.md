# Quality and aligned head occupancy

Question: does raising the aligned head threshold create useful skippable sparsity, and at what loss?

Method and coverage: Both models are fresh random-initialization training, seed1234,712 updates. Only head κ varies (.1/.5); h/z8×16 κ0 and seven-site Ph+head pressure are matched. All500 validation documents yield338 complete2048-token blocks,692224 input tokens and691886 predictions;1444 tail tokens are excluded. Native training evaluation uses B32; the endpoint uses B1 on one RTX5090. The head gate is sparse activation masking; training and the native head baseline use dense matmul. All h/z projections remain native dense in the endpoint comparison. Only custom sparse head execution bypasses MMA.

Sources: [training reduction](../results/training-comparison.json), [endpoint reduction](../inference/results/comparison.json),
and [byte identities](../results/figure-sources.json). Integer activation/tile/instruction counts are pooled before division.
Loss is summed NLL divided by all691886 predictions, not an average of batch losses.

Caption: left, final native validation loss (zero-baseline axis). Right, empty16×16 input tiles under B32 native evaluation
and independently counted B1 custom-consumer MMA bypass. Blue denotes native dense measurements; green/hatching denotes
custom sparse execution. B1 empty gate tiles equal empty consumer tiles and skipped/potential MMA exactly.
Different B32/B1 activation arithmetic can change occupancy; these are reported separately.

Result: head κ.1 loss=5.560725, B32 empty tiles=39.878%, B1 bypass=39.881%;
head κ.5 loss=5.882600, B32 empty tiles=70.763%, B1 bypass=70.763%.
The paired higher-threshold loss delta is +0.321875.

Caveats: one seed, model size and training budget. The two models isolate threshold at fixed alignment and pressure;
historical Run066 checkpoints do not isolate tile alignment alone. Empty tiles/logical products do not establish speed.
No acceptable-quality cutoff was declared. This observation is not a promoted manuscript claim.

Source script: [19_figures.py](../19_figures.py). Output: [PDF](../figures/01-quality-and-aligned-occupancy.pdf).
