# P2 versus Ph: measured inference latency

RTX5090, BF16, B1/T2048, full logits; K050. Lower milliseconds is faster.

| kappa | Ph geomean ms | P2 geomean ms | Paired P2/Ph speedup | P2 latency reduction | Process range | Qualified |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.659690 | 0.664153 | 0.993279x | -0.677% | 0.988689--0.996753 | True |
| 0.1 | 0.625165 | 0.614790 | 1.016875x | +1.660% | 1.015749--1.018378 | True |

Speedup is the geometric mean of paired latency ratios, not the ratio of selected medians.
Each threshold has1344 direct pairs in three fresh processes, with complete338-block numerical qualification.
Process repetitions are systems repetitions, not new training seeds. A failed numerical gate prevents an accepted kernel speedup claim.

| kappa | Execution | Host median ms | CUDA median ms | Input tokens/s from host median |
|---:|---|---:|---:|---:|
| 0 | ph_native | 0.738641 | 0.694432 | 2772659.5 |
| 0 | p2_native | 0.739479 | 0.695616 | 2769517.3 |
| 0 | ph_candidate | 0.645613 | 0.601200 | 3172179.0 |
| 0 | p2_candidate | 0.649803 | 0.605408 | 3151724.6 |
| 0.1 | ph_native | 0.738362 | 0.694208 | 2773707.3 |
| 0.1 | p2_native | 0.740039 | 0.695392 | 2767421.8 |
| 0.1 | ph_candidate | 0.609016 | 0.562752 | 3362801.7 |
| 0.1 | p2_candidate | 0.598679 | 0.553024 | 3420865.1 |

Native denotes the unmodified graph scaffold, qualified against its own eager checkpoint.
Throughput is the inverse batch-1 latency for this uncached full-sequence workload, not autoregressive decoding throughput.
