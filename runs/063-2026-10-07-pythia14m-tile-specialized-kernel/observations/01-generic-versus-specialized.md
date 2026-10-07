# Generic versus corner-specialized inference

Question: does direct consumption of the existing corner8x16 structure lower latency?
Method: paired synchronized host replay; same checkpoints/gate, five graph implementations.
Coverage: RTX5090 BF16 B1/T2048 full 50304 logits;64 inputs x 7 passes x 3 fresh processes per threshold;
all 500 validation documents,338 blocks,1444-token excluded tail. All graphs qualified.
Legend/caption: blue generic Run061 consumer; orange Run063 specialization; bars are pooled geometric mean host ms.
Results:
- kappa=0: 1.117644x; latency change=-10.526%; process ratios=[1.1132388937972924, 1.1126003958255253, 1.1271516805079944].
- kappa=0.1: 1.097651x; latency change=-8.896%; process ratios=[1.1006851043754897, 1.100173829142652, 1.0921158880068993].

Caveats: one trained seed/GPU; unchanged quality penalty of the corner gate; no decoding/DRAM portability claim.

All six full processes produced zero maximum absolute logit error and zero
relative L2 for every graph/block comparison (10,140 comparisons). Pooled BF16
loss matches eager exactly:5.556248486078721 at kappa 0 and 5.5987847543876 at
kappa .1. Numerical bounds were unchanged. Activation and 8x16 tile statistics
also match exactly between the generic and specialized diagnostics.

## No-skip controls and interpretation

Pooled geometric mean synchronized host latency, milliseconds:

| Implementation | kappa 0 | kappa .1 |
|---|---:|---:|
| Native graph |0.871060|0.867335|
| Generic, skipping enabled |0.578467|0.545248|
| Specialized, skipping enabled |0.517577|0.496740|
| Generic, skipping disabled |0.646385|0.643798|
| Specialized, skipping disabled |0.668422|0.674537|

The specialized consumer reduces latency by10.526%/8.896% relative to generic
with skipping enabled. All three fresh-process ratios exceed1 at both doses.
With skipping disabled, the specialized package is 3.409%/4.775% slower.
Thus the observed advantage depends on this sparse workload and skip handling;
the package is not a universally faster dense projection. The comparison does
not isolate corner inspection from warp/block changes, or establish DRAM
traffic savings. Generic no-skip also disables its short-row SIMT substitution;
the specialized consumer has no such substitution.

CUDA-event geometric means support the same direction:0.559860 to0.496341 ms
at kappa 0, and 0.526454 to0.476740 ms at kappa .1. Standalone support-inspection
measurements and integer MMA/SIMT counters are retained in the source attempts;
hot-resident inspection timings are not additive attribution of model latency.

The approved question is supported for these checkpoints and this workload:
consuming the induced corner-tile structure recovers additional measured speed
without changing weights, gates or inference outputs. It does not remove the
trained tile models' validation penalty relative to the scalar controls.

## Execution and provenance

Secure RTX5090 in EUR-NO-1, host CUDA 12.8, pinned application and source bundle.
Tag002 followed an infrastructure-only pinned-wheel download retry; no
scientific settings changed. All 86 returned files and 14,104,050 payload bytes
passed SHA256/byte verification before Pod termination. The six full processes
took 287.748 s, of which timing phases totalled 12.511 s. Pod life was51 min 52.666 s;
transfer, setup, compilation and retries explain the remaining time. Estimated
GPU plus running storage isUSD 0.864, provisional until billing posts. No owned
billable resource remains. See [closeout](../prelaunch/closeout-001.json),
[transfer receipt](../transfer/receipt-002.json) and the two infrastructure retry
records. No manuscript or research-finding promotion is included.
Source script: [21_figure.py](../21_figure.py). Evidence: [results.json](../results/results.json).
Figure: [PDF](../figures/01-generic-versus-specialized.pdf).
Result SHA256: b28581b38699176e7fdc08f4678d45cd415e3d16bebfb237c891692662ff14c9.
