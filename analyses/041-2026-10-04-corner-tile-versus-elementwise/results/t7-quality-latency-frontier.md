# T7 corner endpoints against the orange frontier

| Endpoint | kappa | Canonical FP16 loss | Sparse latency (ms) | Process range (ms) | Nondominated? |
|---|---:|---:|---:|---|---|
| T7/Ph tile | 0 | 5.721071 | 0.564432 | 0.564240–0.564585 | no |
| T7/Ph tile | 0.1 | 5.835102 | 0.516416 | 0.514983–0.518140 | no |

| kappa | Same-session scalar/tile sparse latency ratio |
|---:|---:|
| 0 | 1.125277x |
| 0.1 | 1.077430x |

Ratios >1 mean tiles are faster.
Original orange T7/Ph and prior T2 coordinates are from separate sessions. Run062 adds gate passes included in timing. Same-session controls establish paired runtime ratios; historical Pareto membership is descriptive. One seed; whiskers are process ranges, not confidence intervals.
