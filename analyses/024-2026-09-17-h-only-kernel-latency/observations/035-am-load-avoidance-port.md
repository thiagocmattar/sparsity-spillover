# Does the final h/z strategy help at a/m?

Question: does a direct port of the final h/z load-avoiding hybrid improve a/m,
providing a stronger empirical explanation for the kernel's site dependence?
This follows the [Run037 per-operation analysis](034-operation-latency-manuscript.md)
and the approved [design](../AM-LOAD-AVOIDANCE-DESIGN.md).

Method and coverage: Run039 tests the retained 14M T7/Pall kappa=0.5 checkpoint
on one RTX 5090, with gates, weights and all other paths fixed. Five modes,
three fresh randomized processes each; all 338 validation blocks per process;
64 timing inputs, seven paired passes and all 50,304 output logits. Full details,
raw evidence and caveats are in the [run observation](../../../runs/039-2026-09-19-pythia14m-am-port-fixture/observations/001-am-port-verdict.md).

Table caption: same-session geometric mean full-model host latency, with only
the selected a/m implementation changed. The sparse-logic-off control uses
the new port layout and preserves the model's gates; it is not A0.

| Implementation | Latency (ms) | Increase from frozen |
|---|---:|---:|
| Frozen K050 | 0.494995 | reference |
| Port at a | 0.580284 | 17.23% |
| Port at m | 0.630549 | 27.38% |
| Port at a and m | 0.715227 | 44.49% |
| Both-port layout, sparse logic off | 0.708533 | 43.14% |

All correctness checks pass with zero measured logit and pooled-loss error.
All port process means are slower than every frozen process mean. The port
therefore extends correctly but does not improve this checkpoint's latency.

The mechanism is supported by the full-validation diagnostics. Empty 8x16
tiles occupy only 11.12% of the a grid and 0% of m (versus 8.44%/0% at 16x16).
Only 15.37%/0% of a/m rows have at most two nonzeros, versus 92.75%/99.73%
at h/z. The padded fallback issues 1.895x/2x the original a/m matrix instructions.
Most latency regression persists in the dense-layout control, while enabling
sparse logic adds another 6.694 microseconds. The control does not isolate
padding from every other scheduling/layout difference. Source-level weight
request counts are not measurements of DRAM traffic.

Conclusion: retain the current a/m paths for this tested implementation.
This is evidence against this direct port, not a proof that all a/m load
avoidance must fail or that the selected kernel is optimal. The result concerns
one checkpoint, BF16 full-sequence inference and one GPU type. It does not add
training results or replace the canonical quality/sparsity evaluation.

Sources: Run039 [benchmark](../../../runs/039-2026-09-19-pythia14m-am-port-fixture/02_benchmark.py),
[latency reduction](../../../runs/039-2026-09-19-pythia14m-am-port-fixture/07_reduce.py),
[mechanism reduction](../../../runs/039-2026-09-19-pythia14m-am-port-fixture/12_mechanism.py),
[latency JSON](../../../runs/039-2026-09-19-pythia14m-am-port-fixture/results/am-load-avoidance.json),
and [mechanism JSON](../../../runs/039-2026-09-19-pythia14m-am-port-fixture/results/mechanism.json).
All 13,440 timing samples and transferred files verified locally. Run038 retains
the initial fixture failure; the corrected run used identical candidate CUDA.
The Pod was deleted after recovery, at approximately USD 0.616 GPU cost plus
storage. No figure or manuscript edit was required for this test.
