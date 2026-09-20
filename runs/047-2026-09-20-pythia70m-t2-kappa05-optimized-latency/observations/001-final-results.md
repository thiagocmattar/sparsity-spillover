# Optimized latency of the later 70M T2/Ph endpoint

## Question and method

Measure the missing T2/Ph kappa=.5 endpoint with frozen opt073; repeat Base and kappa=.1 to expose differences between sessions. Native, original port and optimized implementations are paired in every process. One RTX5090, BF16, batch 1, sequence 2,048, full 50,304 logits; three fresh final processes per checkpoint,64 inputs x7 timing passes per process. All 338 complete validation blocks from 500 documents qualify, with 1,444 excluded tail tokens. No training, kernel search or clipping.

## Results

| Checkpoint | Native ms | Original port ms | Optimized ms | Optimized / native Base speedup |
|---|---:|---:|---:|---:|
| Base | 1.656128879 | 3.314144448 | 2.791958084 | 0.593178x |
| T2/Ph kappa=.1 | 1.721252248 | 2.259116355 | 1.959446431 | 0.845202x |
| T2/Ph kappa=.5 | 1.720100231 | 1.695805837 | 1.214368251 | 1.363778x |

All 9 final processes pass all three implementations. Each implementation/checkpoint has 1,344 host timings. The new endpoint optimized process means range 1.213392--1.215153ms. Its canonical training loss 4.838033648 and pooled model-wide logical sparsity 15.42696957% remain unchanged; BF16 qualification loss 4.846477910 is a separate measurement. Its optimized latency is28.390% lower than the original port in this session.

Base native latency is 1.656128879ms here versus 1.611047709ms in Run045 (+2.798%). Optimized Base changes +2.190%; optimized kappa=.1 changes +0.198%. These are descriptive session differences, not calibration factors. Retain the original 26 Run045 values and this new result separately. No cross-session pooling or rescaling.

## Coverage, retention and caveats

All 217 returned artifacts passed size/SHA256 verification; local reduction exactly reproduces the remote summary. Raw timings, per-block numerical gates, both implementations' activation zero/near-zero/RMS/L2 and logical/work counters, weight norms and profiles are retained. Checkpoints and original optimizer/gradient histories remain in their source runs. The full frozen input inventory and three final model copies are local.

GPU UUID: `3c6e480f-4167-afc4-5f2e-ed93cb201c2e`. Pod `e2njhw4wy9d9uo` was deleted after verification; live listing confirmed zero Pods and the pre-existing network volume unchanged. Independent guard cancelled. Runtime about 22 minutes; estimated compute plus allocated storage below USD0.37, not a settled invoice, within the 90min/USD2 cap.

One training seed and workload; process ranges are not confidence intervals. Recipes differ in quality. The optimized implementation includes dense optimizations, so the gain is not pure skipping. Native Base and same-checkpoint native are distinct references. Canonical sparsity is a logical-product opportunity, not runtime work saved.

## Evidence and figure

Run source: `02_benchmark.py`, `03_execute.py`, `06_reduce.py`; source identities in `provenance/source-freeze.json`. Results: `results/matched-grid.json`, `results/local-verification.json`, `results/closeout-verification.json`; transfer inventory: `transfer/inventory-001.json`. Figure and manuscript integration: Analysis030 `01_integrate.py`, `02_plot.py`, observation001. This run produces no separate publication figure.
