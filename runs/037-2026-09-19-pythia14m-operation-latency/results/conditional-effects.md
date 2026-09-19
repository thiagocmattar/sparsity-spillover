# Conditional full-model latency effects

14M T7/Pall, kappa=0.5. Positive saved time means enabling the path helps.

| Path | Instruction bypass (%) | Path off (ms) | Time saved (us) | Difference span (us) | Conditional speedup |
|---|---:|---:|---:|---:|---:|
| a | 8.44 | 0.519449 | -1.63 | [-5.99, +3.84] | 0.9969x |
| m | 0.00 | 0.518717 | -2.36 | [-5.11, +2.73] | 0.9955x |
| h | 94.50 | 0.671148 | +150.07 | [+145.45, +154.39] | 1.2880x |
| z | 99.64 | 0.550038 | +28.96 | [+23.71, +35.69] | 1.0556x |
| qk | 58.00 | 0.518164 | -2.91 | [-7.29, +1.06] | 0.9944x |
| pv | 67.21 | 0.514900 | -6.18 | [-9.02, -2.18] | 0.9881x |

Spans use process extrema, not confidence intervals. These effects do not add.

| Control | Full-model latency (ms) |
|---|---:|
| frozen | 0.518482 |
| full | 0.521078 |
| off | 0.671637 |
| projection | 0.514037 |

Verified 26,880 raw timing samples, all 30 full-validation processes, and all ten diagnostic passes.
Source and exact values: `conditional-effects-audit.json` and `operation-latency.json`.
