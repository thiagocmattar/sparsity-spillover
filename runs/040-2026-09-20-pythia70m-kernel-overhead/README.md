# Run040: diagnose and reduce 70M kernel overhead

Status: launch approved under the USD6/four-GPU-hour cap; provisioning next.
The user approved both checkpoints in the nine-mode diagnostic on20 September
2026 and additionally authorized optimization toward native-PyTorch speedups.
The [design](../../analyses/024-2026-09-17-h-only-kernel-latency/70M-OVERHEAD-DIAGNOSTIC-DESIGN.md)
and [saved-evidence audit](../../analyses/024-2026-09-17-h-only-kernel-latency/observations/042-70m-overhead-audit.md)
separate this question from the earlier specialized-base speedup comparison.
The user explicitly approved the [launch packet](LAUNCH.md); no additional
confirmation is required within this envelope.

## Fixed science

Retained final step712 Pythia-70M T0/P0 and T7/Ph kappa0.5, original Run035
identities c00/c21. No training, weight changes, gate changes or altered
tolerances. Six layers, width512, intermediate2048, eight64-dimensional heads.
T7/Ph retains one-sided a,m,h,z and symmetric post-RoPE q,k,v gates; training
used h-only OL1 with lambda=b=1. Original initialization/data-order seed1234,
training budget and optimizer are inherited unchanged through the checkpoint.

RTX5090, BF16, batch1, sequence2048, all50304 logits, no cached decode. Native
SDPA and the frozen shape port share the graph scaffold and input staging.
Runtime seed2801; timing seed2504. The full validation remains all500 documents,
338 complete blocks,692224 input tokens,691886 prediction tokens,1444 excluded
tail tokens. Bounds: logit atol0.25+rtol0.02, relativeL2<=0.02, loss delta<=0.001.

## Execution and implementation

`01_prepare.py` verifies1404 immutable input/source copies, including only
the two selected checkpoints, validation, the original training development
cache, pinned Python dependencies and the frozen70M CUDA sources. The original
runs remain unchanged. `controls.py` implements native/full, all-skips-off,
hz-skips-off, and native replacements for h/z, a/m, attention, norm and RoPE.
Native replacements preserve gates and separately rounded branch/residual sums.

`02_benchmark.py` pairs each candidate with its same-checkpoint native graph
and, except for full itself, an untouched full graph in the same process.
Three resident models require more memory than Run035; the32GB GPU has ample
headroom relative to the prior3.44GiB two-model peak. Actual memory is recorded
and checked during smoke. Timing is64 inputs x7 passes x3 fresh processes.
Compilation, layouts and graph capture are excluded; recurring checks and
preprocessing are timed. `03_execute.py` records randomized order for18 smokes
and54 complete diagnostic processes. It reserves20 minutes before the hard
deadline for retrieval. Failures stop the dependent matrix and remain visible.

`10_test_operators.py` reuses the frozen CUDA primitive checks.
`06_control_checks.py` compares all nine controls against native outputs on
four training blocks for both models (72 full-output checks). Smokes use
four timing inputs, two passes and eight correctness blocks. These GPU checks
are pending launch; CPU tests do not substitute for them.

`diagnostics.py` runs after timing in replicate1, pooling actual-operand zero,
near-zero, RMS/L2, row-occupancy and logical-product counts over full validation.
It checks instrumented h/z counts against an operand oracle. It retains
attention instruction counts when that custom path exists; native instruction
counts are explicitly unmeasured, never replaced by sparse-opportunity counts.
`profiling.py` separately records native/full graph and eager traces on four
declared timing inputs in each replicate. Eager traces retain operator shapes
and correlations, graph traces actual kernels. Profiled times are diagnostic.

`07_reduce.py` verifies raw sample pairing, full coverage, one physical GPU,
and three processes per cell. It reports conditional replacement effects and
one common native T0/P0 denominator across both checkpoints. These effects
are nonadditive. Same-checkpoint native and frozen comparisons are separate.

## Approved optimization phase

Use measured bottlenecks to develop at most12 immutable candidates, beginning
with replacement/composition of inefficient paths and then the relevant
inspection, fallback or matrix schedule. Each candidate has its own folder,
code and hashed manifest, recorded by `11_candidate.py register optNNN`.
Use one execution policy for both checkpoints; dispatch may depend on tensor
shape/operands, never checkpoint or validation-input identity. Thresholds and
weights remain unchanged. No promise of a positive speedup is assumed.

Development uses only the first16 blocks of the retained training development
cache, five timing passes, and the same numerical bounds. Run a candidate for
each checkpoint with `02_benchmark.py --candidate optNNN --development ...`.
Choose by T7/Ph development latency among candidates qualified on both; also
report base-model behavior. `11_candidate.py freeze optNNN` records training
evidence and locks one selection before full validation. The benchmark rejects
unselected candidates in final validation. Report failed and successful
development trials, not just the final winner. Candidate-specific instruction
instrumentation must be checked if its matrix layout changes.

Final optimized evaluation repeats both checkpoints x3 processes with all338
blocks and the original64x7 timing, paired with native and untouched full.
Freeze the implementation before those measurements; do not tune to final
validation latency. Numerical failure is retained explicitly. No full kappa
grid or manuscript rewrite is part of this run.

## Recovery and verification

`08_collect.py` inventories raw timings, validation records, diagnostics,
profiles, failures, candidate sources, runtime/source identities and reductions.
`09_verify_retrieval.py` checks archive and per-file SHA256 identities before
accepting data locally. Original final checkpoints remain both local and in
the input inventory; do not terminate the Pod until recovery verifies.
Compilation caches and environments are not scientific outputs.

CPU verification: eight focused tests pass (gate equality, BF16 additions,
native norm gate restoration, partial-RoPE layout/gates, complete54-process
matrix, retained cohort/coverage, common native-base denominator and duplicate
timing-sample rejection). Full bootstrap:242 pass using an isolated
temporary directory. The initial bootstrap invocation had11 setup errors
because the repository's default temporary directory was inaccessible; no
scientific test failed, and the isolated rerun passed all242.

The launch packet supplies live price/availability, ETC, cost ceiling,
storage/transfer, monitoring and the stop guard. No main.pdf rebuild.
