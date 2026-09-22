# Run050: moderate-kappa 70M kernel families

Status: approved design, launch and GPU-period extension; implementation and
execution in progress. Authorization: user message on 22 September 2026,
"Approved the design, launch and GPU period extension."

The approved design is Analysis036 at commit9d6822d8. Test c00 Base and c24/c25
T2/Ph at kappa .05/.1, unchanged retained step712 checkpoints, seed1234,
BF16 B1 T2048 full50304 logits on the retained RTX5090. No training or pruning.
Four bounded families: packed streaming, masked-load split-K, gathered
tensor-core SpMM, exact2:4 routing with lossless dense overflow. Strengthen
the fused-gate dense control and select h/z per layer on training inputs only.
The detailed scientific contract, diagnostic inventory, acceptance bounds,
selection, full338-block validation, 64-input/7-pass/3-process timing and
interpretation limits are preserved in the approved Analysis036 README.

Run049 sources, checkpoints and caches remain immutable dependencies verified
by hashes. All new sources, attempts, failures and observations belong here.

Lease continuation starts2026-09-22T18:42:28Z and ends22:42:28Z (19:42:28
America/Sao_Paulo): four additional GPU-hours, live existing Pod rate$0.99/h,
at most$3.96 additional GPU time plus retained storage. Reuse Pod
kym4fmsrbsg1s6, 30GB container and40GB persistent /workspace; no new volume.
Provider-stop guard replaces the old20:30:06UTC guard after the new guard is
confirmed armed. The Pod is not deleted at completion. Work survives terminal
disconnects; monitor every60seconds, reserve20minutes for retrieval, and verify
artifact hashes locally. User has already approved both execution and this
extension; routine checks do not require another approval request.

## Implementation and retained attempts

`01_prepare.py` pins literal canonical training-cache blocks0:128. The older
Run049 development sample is not that literal prefix, so it is not silently
reused. Checkpoint and validation identities remain unchanged. `02_operator_checks.py`
tests zeros, dense inputs, sparse inputs, threshold equality, overflow and
changed-input CUDA graphs. `03_screen.py` compares whole conversion-plus-compute
paths on every training block. `04_select.py` uses matched within-screen ratios;
it never compares raw milliseconds between separate component-screen sessions.
`05_benchmark.py` and `06_execute.py` handle fresh-process full-model qualification
and paired timing. `07_diagnostics.py` retains the agreed full-validation inventory.

- operators-001 stopped in an inherited no-symlink packaging check before GPU
  tests. The declared Run049 archive cache symlink is valid; the retry verifies
  its byte sizes and hashes instead. No scientific input changed.
- operators-002 passed all A/B/C and dense-control primitive cells. Family D's
  initial PyTorch API cannot return FP32 from BF16 sparse multiplication.
- operators-003 retained a missing CUTLASS include-path compile failure.
  operators-004 uses the existing archived dependency and passes every D cell,
  including exact overflow conservation and independent FP32 result checks.
- screen-abc-001 completed both kappas,128 training blocks each. A and C passed
  the predeclared refinement criterion; B did not. Shared initial policies use
  sparse h in layers1--4 (zero-based) and dense elsewhere.
- screen-d-001 stopped with SIGBUS at c24 block122. The deployment helper had
  rewritten the unchanged memory-mapped training file during that process.
  This is the identified infrastructure cause; its partial screen is excluded.
  Deployment now skips identical files, forbids changed immutable inputs and
  replaces source files atomically. screen-d-002 is the unchanged-input retry.
- training-abc-001 completed all three checkpoints and128 training blocks each.
  Every integrated implementation passed the logit and pooled-loss bounds.
  These are training measurements, not final validation evidence.

Six authorized refinements target the two eligible families: C row groups1/2
with K32, C N128 tiles for row groups4/8/16, and A eight row warps with V4.
This stays within the approved eight-refinement ceiling. All gates, weights,
packing, conversion, reduction and output rounding retain the approved contract.

The full bootstrap suite and initial run-local checks passed together (245
tests). Two further focused checks validate cross-screen selection and runtime
union-counter conservation (five run-local tests now pass). Source/deployment
archives and all rejected attempts are retained; no manuscript edits are made.

## Final preparation

The refined shared C policy uses sparse h layers1--5, with row groups8/4/4/4/16
and N128/K32 tensor-core tiles; h layer0 and every z layer stay dense. The A
alternative uses sparse h layers1--4. Whole-model training-refined-001 passes
all128 training blocks at all three checkpoints, including the C skip-disabled
ablation. Skipping can be disabled without changing gates, weights, row layout
or conversion work; that ablation traverses every K feature in the consumer.
Initial and refined training measurements remain separate, provisional evidence.

operators-006 passes440 primitive cases and88 changed-input graph checks,
covering24 initial configurations, six refinements, three dense controls and
eleven untuned C skip-disabled ablations across both K shapes. The final CPU
suite passes248 tests (242 bootstrap and six focused run tests). Diagnostics
smoke002 covers four training blocks at all four integrated policies, with
counter conservation, all-site activation statistics, compiler artifacts and
separate graph/eager profiles. Smoke001 retained an archived-module name
collision; explicit Run049 helper bindings fix it before any final validation.

The deploy-history audit found three local snapshot-file races in two archives.
Remote deployed files had passed their recorded hashes. Exact source bytes were
recovered and checked against those original SHA256 values, with the original
archives retained. `provenance/source-history-audit.json` records the recovery.
The sender now uploads and archives the same immutable in-memory byte snapshot.

`10_freeze.py` seals the training-selected policy and checks executed kernel
identities. `12_complete.py` runs nine final processes, full-validation diagnostics
and four bounded hardware-counter probes sequentially under the absolute lease.
`11_reduce.py` reports paired input-block intervals with process strata and a
hierarchical sensitivity interval; it cannot change the frozen policy.
