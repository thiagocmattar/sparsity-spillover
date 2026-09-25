# Interpretation of the conditional mechanism ablation

Question: which execution mechanism provides a net full-model latency benefit
for the retained 14M T7/Pall kappa=0.5 checkpoint?

Method and coverage: the approved Run056 factorial, all 15 fresh processes on
one RTX 5090, all 338 validation blocks per process, and 64 fixed timing inputs
with seven paired passes. See [001](001-mechanism-ablation.md) for the figure
caption and [the timing table](../results/mechanism-effects.md) for process
ranges. All four implementations and the frozen anchor have validation loss
5.83130066301001. Candidate/frozen logits and h/z operands are bitwise equal.

Tile bypass alone is the fastest measured configuration (0.494815 ms), followed
closely by both mechanisms (0.496476 ms). Both save 165.031 us, or 24.95%, versus
neither (0.661507 ms). The padded custom fallback is the comparator for that
percentage; it is not an optimal dense implementation or the PyTorch baseline.

Conditional on short-row execution, tile bypass saves 82.851 us (14.30% of the
short-row-only time), with a process-extrema difference span of 80.941 to
85.017 us. Conditional on tile bypass, short-row execution costs 1.661 us
(0.34% of the tile-only time), with a slowdown span of 0.331 to 3.066 us.
These spans are descriptive, not confidence intervals. The small slowdown is
observed on this device/checkpoint/workload; it is not a universal claim about
scalar execution.

Short-row execution does help when tile bypass is disabled: it saves 82.180 us.
The interaction is -83.841 us, showing substantial overlap between the two
standalone benefits. A unique percentage allocation of the joint saving to
each mechanism is therefore not justified.

The [work counters](../results/mechanism-work.md) confirm that the short-row
path executes and changes work. With both mechanisms, 6,409,724 zero rows use
bias-only completion, 1,281,723 one-nonzero rows and 302,911 two-nonzero rows
use scalar arithmetic, and whole short-row groups omit 271,613,824 padded
matrix instructions. Issued matrix instructions fall from 32,461,488 for tile
only to 14,862,352 for both; this extra reduction does not translate to a net
latency gain. Counters establish the work change, not its hardware-level cause.
They do not separate inspection, routing, scalar arithmetic, scheduling, memory
access, and matrix execution costs. Weight-request estimates are source-level
BF16 element requests, not measured DRAM traffic.

The rewritten both-enabled control and frozen K050 differ in geometric-mean
latency by only 0.00468%, and their process ranges overlap. All output checks
pass, and the two uninstrumented h/z kernels have identical reported resources:
43 registers, 5,200 shared bytes, zero stack/local bytes. This satisfies the
prespecified fidelity review rule; it is not a statistical equivalence test.

The supported conclusion is a conditional one: tile bypass provides a clear
net benefit here, while the existing short-row path supplies no additional net
benefit once tile bypass is present. The experiment does not isolate scalar
arithmetic from zero-row/group completion, nor does it attribute the entire
PyTorch speedup to either mechanism. The manuscript remains unchanged pending
the human's decision about how to use this finding.

Sources: `07_reduce.py`, `13_report.py`, `15_work_summary.py`, the retained raw
timing/validation/diagnostic files, and `runtime/compiler/{t11,frozen}.txt`.
