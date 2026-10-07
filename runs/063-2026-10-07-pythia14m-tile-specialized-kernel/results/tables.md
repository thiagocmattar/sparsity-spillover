# Generic versus corner-specialized consumer

RTX5090 BF16 B1/T2048 full50304 logits; uncached; equal staging/setup excluded; six fresh processes

| kappa | Generic sparse ms | Specialized sparse ms | Speedup | Latency reduction | Process speedups | BF16 eager loss |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 0.578467 | 0.517577 | 1.117644x | 10.526% | 1.113239, 1.112600, 1.127152 | 5.556248 |
| 0.1 | 0.545248 | 0.496740 | 1.097651x | 8.896% | 1.100685, 1.100174, 1.092116 | 5.598785 |

All five graphs qualified against their own eager checkpoint on all 338 validation blocks.
Same weights, corner gate and input coverage. Three processes are runtime repetitions, not independent training seeds.
Both no-skip controls and CUDA-event samples are retained in results.json.
