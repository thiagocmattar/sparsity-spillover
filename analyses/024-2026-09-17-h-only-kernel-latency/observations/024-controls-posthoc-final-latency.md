# Dense/ReLU control clipping: measured final-kernel latency

PDF: [07-controls-posthoc-final-latency.pdf](../figures/07-controls-posthoc-final-latency.pdf).
All values: [Run036 latency table](../../../runs/036-2026-09-18-controls-clipping-final-kernel/TABLE.md).

## Question and method

What latency accompanies the retained post-hoc dense/GeLU and ReLU/1-site
clipping sweeps at 14M and 70M under their final kernels?
Run036 reuses four unchanged step712 checkpoints and the exact Run030
thresholds for p=0, 0.1,...,0.9. Clipping zeroes abs(x)<=t at a,m,h,z in all
six layers, after the trained activation. ReLU/1-site describes the trained
activation topology; its post-hoc clipping still covers all four sites.

The frozen kernels are K050 at 14M and the qualified k050-70m-v2 port at 70M.
PyTorch clipping masks execute inside the timed CUDA graphs; zero-dose identity
masks are elided. Each point pools 1,344 paired synchronized host timings from
three fresh processes, 64 fixed validation inputs and seven passes, on one
RTX5090. Execution is BF16, batch 1, T=2048, uncached, with all 50,304 logits.
Native graph timing uses the same clipping rule. Compilation, setup and equal
input staging are excluded. The randomized process order is retained.

## Coverage and verification

All 40 settings and all 120 full-validation processes qualify under the unchanged
logit and loss bounds. Each process uses all 338 complete 2,048-token blocks
from 500 MiniPile validation documents: 692,224 input tokens, 691,886 shifted-label
prediction tokens and 1,444 excluded tail tokens. The output archive and each
of its 1,032 inventoried files were hash-verified locally before Pod deletion.
The 12 smoke settings are separate from final timing aggregation.

Retained FP16 clipping losses and integer-pooled logical sparsity supply the
quality and sparsity axes. New BF16 native/candidate losses, activation
exact/near-zero counts, RMS/L2, weight norms, kernel skip counters, occupancy,
thresholds, raw timings and checkpoint/cache identities remain available.
The BF16 scalar opportunity diagnostic is explicitly a lower bound and is
separate from canonical FP16 logical sparsity.

## Caption and legend

**Post-hoc Clipping Sparsity and Latency.** Panels (a)-(c) show 14M and panels
(d)-(f) show 70M; columns show latency versus clipping target p, retained
model-wide logical sparsity, and retained FP16 validation loss. Gray circles
denote Base model (GeLU); olive circles denote ReLU. All 40 points and the full loss range
are visible. Labels identify p=0,0.5,0.9 on the quality-latency curves. Latency
includes recurring clipping work. Axes share latency limits within each size.
Solid lines connect inference settings of one fixed checkpoint. The jump from
p=0 to positive p includes the added mask operators. Target p is not achieved
model-wide sparsity. The native comparison and process ranges are in the table
and source JSON. This figure is separate from the six-panel all-minus-h
pressure contrast in Figure3.

The 12.8-by-5.8-inch layout, 14-point title, 11.5-point panel titles, 11-point
axis labels and legend, 10-point ticks, 1.6-point lines, circular markers, and
light horizontal grid match Figure3. Control colors remain gray and olive.
Axis labels omit precision details for readability; the retained FP16 versus
runtime BF16 distinction remains part of the measurement protocol above.

## Implementation note for later text

The final kernels exploit activation sparsity at matrix-instruction granularity.
QKV and FFN-up bypass a BF16 16-by-8-by-16 matrix multiply-accumulate instruction
only when its entire 16-by-16 activation tile is zero. FFN-down and
attention-output use groups of eight token rows and 16 input features, with
the eight active rows padded to the same 16-by-8-by-16 instruction shape.
Rows with at most two nonzero activations can instead use a short scalar path,
subject to the kernel's numerical safety checks; the remaining rows determine
whether each 8-by-16 activation group requires matrix instructions. Reported
bypassed instructions therefore include scalar substitution, whose products
are counted separately. The weights remain dense.

Post-hoc magnitude clipping is applied at a,m,h,z in all six layers using
separate PyTorch operators inside the timed CUDA graph; it is not fused into
the frozen kernels. At p=0 the value-identity clipping operators are omitted.
Moderate clipping can increase scalar zero-product opportunity without creating
entirely zero activation tiles or sufficiently short rows. Consequently, the
number of matrix instructions can remain unchanged as logical sparsity rises.
The vocabulary head remains dense, and the retained QK/PV instruction counters
are unchanged across these clipping settings. Any benefit must also cover
clipping, sparsity inspection and the remaining execution costs. This study
does not isolate the duration of those individual costs.

For a concrete retained example, the 14M Base model at p=0.5 has approximately
49-55% BF16 activation zeros at the clipped sites but bypasses no projection
matrix instructions. At p=0.9 it bypasses 9.73% of FFN-down and 98.72% of
attention-output matrix instructions. Its latency falls from 0.787646 ms at
p=0.5 to 0.718922 ms at p=0.9, while remaining above its unmodified p=0 latency
of 0.659841 ms. Thus high-p skipping is useful to this execution path, but it
does not establish a net improvement over the unclipped 14M control or a
quality-preserving speedup. These observations concern the frozen implementation,
not an intrinsic limit on other kernels for unstructured sparsity.

Implementation provenance: [operand and instruction counter definitions](../../../runs/036-2026-09-18-controls-clipping-final-kernel/diagnostics.py),
[timed clipping adapter](../../../runs/036-2026-09-18-controls-clipping-final-kernel/clipping_adapter.py),
and the per-operation counters and timings in the
[verified reduction](../../../runs/036-2026-09-18-controls-clipping-final-kernel/results/clipping-final-kernel.json).

## Result

| Size/control | p=0 latency (ms) | p=0.9 latency (ms) | p=0.9 speedup over own p=0 | FP16 loss: p=0 to p=0.9 |
|---|---:|---:|---:|---:|
| 14M dense/GeLU | 0.659841 | 0.718922 | 0.9178x | 5.208594 to 8.825610 |
| 14M ReLU/1-site | 0.639444 | 0.709959 | 0.9007x | 5.269633 to 8.869450 |
| 70M dense/GeLU | 3.340342 | 2.712342 | 1.2315x | 4.099766 to 9.179397 |
| 70M ReLU/1-site | 3.317584 | 2.807996 | 1.1815x | 4.222750 to 9.133301 |

Every positive-p 14M point is slower than its own unmodified final-kernel
control. At 70M, p=0.8 and0.9 reduce final-kernel latency, with substantial loss
increases; ReLU p=0.7 has only a marginal 1.0023x point estimate. All 70M
final-kernel points remain slower than the matched clipped native graph.
For example, at p=0.9 the native graph takes 1.861467 ms for dense and 1.842973 ms
for ReLU. A gain over the same final kernel at p=0 is therefore different from
a gain over the native implementation.

## Caveats and provenance

The results apply to these frozen kernels with timed PyTorch clipping adapters.
FP16 quality/counts and BF16 runtime masks are separate measurements; the JSON
retains BF16 losses as well. Process minima/maxima are not confidence intervals.
One training seed, one GPU and one timing session do not support independent-seed
or cross-device uncertainty claims. The70M port has its established shape and
optimization-budget limitations. No decoding/KV-cache workload is measured.
Logical zero products and kernel skip counters are not measured speedups.

Source scripts: [Run036 reducer](../../../runs/036-2026-09-18-controls-clipping-final-kernel/16_reduce.py)
and [figure builder](../15_plot_controls_clipping_latency.py).
Source data: [verified 40-point reduction](../../../runs/036-2026-09-18-controls-clipping-final-kernel/results/clipping-final-kernel.json),
[figure evidence and hashes](../data/controls-clipping-latency.json), and
[run record](../../../runs/036-2026-09-18-controls-clipping-final-kernel/README.md).
The PDF was rendered and visually checked; both fonts are embedded.
No manuscript TeX or promoted finding was changed.
