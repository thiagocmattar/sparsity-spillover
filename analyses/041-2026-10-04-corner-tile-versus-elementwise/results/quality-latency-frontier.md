# Corner-tile endpoints in the previous quality-latency frontier

| Gate | κ | Canonical FP16 loss | Run061 sparse ms | Process mean range (ms) | On observed frontier? |
|---|---:|---:|---:|---|---|
| corner_tile | 0 | 5.553770 | 0.540220 | 0.539432–0.541275 | No |
| corner_tile | 0.1 | 5.596746 | 0.508340 | 0.506742–0.510481 | No |
| elementwise | 0 | 5.127033 | 0.606922 | 0.606780–0.607099 | Reference only |
| elementwise | 0.1 | 5.152346 | 0.565242 | 0.563694–0.567941 | Reference only |

Historical T2/Ph κ=.5: loss5.536255, latency.506323ms. It dominates both tile endpoints by the retained coordinates.
Historical and Run061 sessions/kernel variants differ. Raw coordinates are not normalized. Frontier is descriptive, not a same-kernel causal comparison or significance test. Auxiliary scalar references are excluded from frontier membership.
