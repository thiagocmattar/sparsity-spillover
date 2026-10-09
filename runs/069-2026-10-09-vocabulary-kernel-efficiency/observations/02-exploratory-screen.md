# Exploratory projection implementation screen

Question: which bounded implementation changes reduce the custom head penalty,
and which proposed remedies fail before final model qualification?

Method: one screen process per retained checkpoint on the same RTX 5090. Eight
variants each run with MMA bypass disabled/enabled; native dense GEMM is the control.
Gate, FP32 certificates, FP16 operands/output, accumulation and weights stay fixed.
No clipping, metric change or new training occurs.

Coverage: 64 deterministic real operands per checkpoint and seven randomized
paired passes = 448 timing samples/mode. All candidates pass unchanged numerical
checks on those operands. Separate untimed Count specializations retain issued,
skipped and potential MMA counts; they are absent from timed graphs. The screen
is exploratory and does not itself provide full 338-block qualification.

Figure: [02-exploratory-screen.pdf](../figures/02-exploratory-screen.pdf).
Orange means custom bypass disabled; green means enabled; the dotted blue line
marks the native dense-GEMM host latency. Colors match figure 01. All modes use
the same trained head gates. Legacy is the original kernel at the stated N;
direct changes stores, prefetch changes operand scheduling, and reuse2 shares
weight loads over two independent M16 row tiles. N is vocabulary tile width.

Result: at N128, reuse2 reduces no-skip latency about 34% relative to legacy N128.
Direct N256 is about 51% slower than legacy N256 without skipping. Width N512
helps modestly; prefetch does not close the dense gap. The selected families are
qualified separately in figure 01, using three fresh processes per checkpoint.

Caveats: the row-reuse intervention also changes row work per block, CTA count and
resource allocation; its effects cannot be separated by this ablation. Source-derived
weight-request reductions are not measured DRAM traffic. Counter access was denied.
The synthetic zero-MMA diagnostic is not a runtime lower bound or a quality result.
Screen rankings are not a global quality-latency frontier or a universal winner.

Source: [raw screen](../artifacts/screen/), [binary resources](../prelaunch/resources-cloud.json),
[diagnostic data](../results/diagnostic.json), and [figure provenance](../figures/provenance.json).
Generator: [14_figures.py](../14_figures.py). Interpretation: [13_diagnose.py](../13_diagnose.py).
