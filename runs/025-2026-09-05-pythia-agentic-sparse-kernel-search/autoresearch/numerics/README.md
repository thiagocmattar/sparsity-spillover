# P0 numerical diagnosis (candidate-only)

The original kernel, adapter, config, evaluator, and calibration evidence remain
unchanged. This folder does not contain an accepted kernel revision.

## Working hypothesis and falsification

P0 serially accumulates BF16 products in FP32 in ascending nonzero-coordinate
order, then adds the bias before BF16 rounding. The FP32 fused GEMM oracle uses
a different reduction schedule. Small local rounding differences can change
downstream gate decisions or residual/attention computations and yield a small
number of logits beyond the predeclared elementwise tolerance.

Evidence supporting investigation, not yet proving this explanation:

- The preserved A1-H-14M diagnostic reproduces the failure on validation block 1:
  P0 has 26 violating logits, relative L2 0.001759, and maximum tolerance excess
  0.03961. The adapter-dense path is bitwise identical to native.
- The FP32 fused oracle passes the two tested blocks but is not bitwise native.
  It has **not** passed complete validation.
- Splitting BF16 rounding before bias addition substantially worsens agreement;
  it is not a justified correction.
- Primitive wire-format equality and sanitizer checks passed on both pilot GPUs.
  Those checks do not cover every trained activation scale or full-network
  amplification.

`diagnose_layers.py` executes native forwards and, at each of the four linear
sites, evaluates P0, BF16-native repeat, and FP32 fused GEMM on exactly the same
native input. It records actual execution order, input preservation, sparsity,
output comparisons, and FP64 scalar-dot references for up to eight largest
P0/native discrepancies. It then checks native repeat, adapter-dense, P0, P0
with cloned projection outputs, and the fused oracle at the fixed logit gate.
An optional prefix sweep replaces progressively more projections in observed
execution order to expose error propagation. None of these diagnostic timings
is a performance result.

Interpretation:

- Native-repeat or adapter-dense disagreement refutes an accumulation-only
  diagnosis and points to mutation/nondeterminism/instrumentation problems.
- Changes caused by cloning each adapted output implicate buffer aliasing.
- Same-input P0 discrepancies concentrated around BF16 rounding boundaries,
  with differing reduction references, support numerical propagation.
- Large same-input discrepancies, input mutation, or disagreement with the
  scalar references beyond ordinary rounding warrant revisiting kernel logic.
- A prefix sweep is diagnostic, not a guarantee of monotone error: replacing
  another operation can either amplify or cancel earlier errors.

## Playbook

Run inside the already pinned Run 025 environment; retain CUDA and virtualenv
`bin` directories on PATH for extension compilation. Input hashes are verified.

```bash
python runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/autoresearch/numerics/diagnose_layers.py \
  --condition 14m/a1h --blocks 1 --prefix-sweep \
  --attempt a1h-layer-diagnostic-001 --seconds 600
```

The script requires a new artifact directory, persists each operation and
forward comparison, and checks a ten-minute cooperative limit by default.
The caller should add its standard process timeout and existing lease guard;
this diagnostic does not launch a GPU or provide infrastructure teardown.

Full-model forwards: one native instrumented pass plus five comparison passes,
and 24 more if the 14M prefix sweep is requested. Additional per-layer isolated
calculations and initial compilation are outside that count. Wall time must be
measured on deployment; no ETC or performance claim is inferred from CPU tests.

Passing these selected blocks is **not** acceptance: any candidate still must
pass the original primitive gates and all 338 validation blocks, including the
unchanged absolute complete-validation loss-difference tolerance.

CPU verification: `python -m pytest -q <this-folder>/test_diagnose_layers.py`.
These helper tests cover tolerance counts, sampled dot indexing, signed/zero
operands, bias treatment, and input preservation; they do not execute CUDA.

Provenance: the source failure is
`../../retrieved/rtx5090-001/artifacts/rounding-diagnostic-001/results.json`.
The prior Run README's manuscript paths `draft/sections/06_results.tex` and
`07_discussion.tex` are now archived; the current inference positioning is
`manuscript/draft/related-work.tex`. This diagnostic changes no manuscript text.
