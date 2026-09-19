# a/m direct-port result

14M T7/Pall, kappa=0.5. All timings measured in the same session.

| Mode | Full-model latency (ms) | Process range (ms) |
|---|---:|---|
| frozen | 0.494995 | 0.492736 to 0.497663 |
| port-a | 0.580284 | 0.580026 to 0.580607 |
| port-m | 0.630549 | 0.629739 to 0.631037 |
| port-am | 0.715227 | 0.713795 to 0.716800 |
| port-am-dense | 0.708533 | 0.706923 to 0.709426 |

| Comparison | Saved time (us) | Speedup | Difference span (us) |
|---|---:|---:|---|
| port-a | -85.290 | 0.85302 | [-87.870, -82.363] |
| port-m | -135.554 | 0.78502 | [-138.301, -132.075] |
| port-am | -220.232 | 0.69208 | [-224.064, -216.132] |
| port-am-dense | -213.538 | 0.69862 | [-216.690, -209.260] |
| port_sparse_effect | -6.694 | 0.99064 | [-9.877, -4.369] |

Maximum recorded absolute logit error: 0.0.
Logical counts unchanged: True.
Positive saved time means faster than frozen, except port_sparse_effect uses the port-dense control.
