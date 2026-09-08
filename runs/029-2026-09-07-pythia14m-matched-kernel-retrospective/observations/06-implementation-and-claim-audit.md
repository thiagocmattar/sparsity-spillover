# Implementation audit and admissible paper claims

## Question and verdict

On 8 September 2026 the user requested an implementation/figure audit,
followed by a short manuscript section if the scientific interpretation was
supported. The audit supports a **scoped systems case study**, not the three
claims in their unrestricted form. No identified error invalidates the frozen
35-checkpoint K050 result or the plotted values. This is not a certificate
that every historical implementation is correct or free of bugs.

| Proposed interpretation | What the evidence supports | What it does not establish |
| --- | --- | --- |
| A GPT-6-Astra coding agent can develop high-performing sparse transformer kernels | One human-guided Codex development trajectory produced qualified Pythia-14M full-model gains, including a positive sparse-path contribution; `gpt-6-astra` is the recorded configured default | Per-request served-model identity, autonomous development, agent superiority, state-of-the-art performance or broad transformer generalization |
| Given specialization, speedup scales with R_model | Positive checkpoint-level association for this fixed K050 workload: OLS R_squared 0.781257 | A strictly monotonic, proportional, universal or causally identified speed law; there are 41 inverted checkpoint pairs |
| Training-conditioned sparse responses have inference potential | These trained checkpoints contain exploitable activation sparsity; sparse paths add measurable acceleration under the declared conditions | Preserved task quality, universal benefit from more sparsity, transfer to cached decoding or other sizes/devices, or that all gains come from zero skipping |

## Coverage and verification method

`19_audit_evidence.py` writes the new, independent
`results/implementation-audit-001.json`; it does not overwrite benchmark
artifacts or invoke the old reduction scripts' output-writing entry points.
The audit verifies:

- 5,175 file identities, including 1,374 archived files, the frozen phase
  sources, checkpoint/data inputs, result/quality/timing records, iteration
  map and current figure source/PDF;
- all 50 historical candidate identities, their eligibility/exclusion
  records and source hashes; 79 candidate Python files parse successfully;
- all 1,173 scheduled outcomes and their candidate/condition/replicate/phase
  identities, with no smoke results substituted;
- all 522,816 native/candidate timing pairs, full `[1,2048,50304]` output
  shapes, exact 64-input identities, seven passes and three fresh processes;
- all 391 grouped comparisons and their raw-pair geometric means;
- qualification reconstructed from all stored per-block gate statistics and
  pooled losses, with identical native loss across candidate processes at
  each checkpoint and a qualified native graph throughout;
- canonical integer count pooling, causal/full-head architecture denominator,
  qualified fixed-c30 incumbent selection, independent least-squares fit,
  ablation ratios and equality of the current PDF with its verified revision3.

Outcomes reconcile exactly: 855 qualified processes, 312 numerical failures
and 6 unsupported. History contains 140 qualified and 74 numerically failed
timed comparisons, plus two unsupported combinations. K050 final coverage
is 35/35 qualified checkpoints (105/105 processes).

Semantic source review focused on the shared benchmark/qualification code,
canonical counters, early adapter families, the executed final dependency
chain, and its modified FlashAttention diff: graph input refresh, mask/causal
semantics, gate placement, residual and BF16 rounding, complete output
coverage, warp-uniform bypass, hybrid-row buffers and diagnostic counters.
The K050 prefix shortcut is disabled. Sparse checks occur during forward;
only static weight layout preparation is excluded from timing.

The audit does **not** freshly compile/rebenchmark all GPU candidates, run
Compute Sanitizer, inspect every vendor-library line, or prove correctness
on arbitrary tensors. Eight excluded proposals are not verified 14M full-model
implementations. All eligible installation interfaces are covered by the
existing suite, with the Windows Triton case explicitly skipped and its
earlier Linux/GPU result retained. Full historical logits were not saved;
re-audit of elementwise gates relies on retained block statistics and the
reviewed checker, not newly regenerated logits. These are verification limits,
not undisclosed claims of fresh GPU testing.

## Bugs and limitations found

### Non-cohort fused-threshold rounding bug

The inherited K019 RoPE/gate CUDA code and K049 h/z CUDA gate compare BF16
values promoted to FP32 against an FP32 threshold. PyTorch's wrapped scalar
comparison rounds that scalar to BF16. For `kappa=0.7`, the BF16 value
`0.69921875` survives the native gate but is removed by the fused comparison.
The source accepts such a threshold, so this is a genuine portability defect,
not merely a hypothetical unsupported shape. K050's a/m gate normalization
already rounds explicitly and is not affected by this particular defect.

`tests/test_audit.py` reproduces the mismatch on CPU and exhaustively compares
the gate masks over all 65,280 finite BF16 values for both one-sided and
symmetric gates at every actual checkpoint threshold. The verified cohort
uses only `0, 0.01, 0.05, 0.1, 0.5`, all of which agree. Thus this defect does
not alter any plotted checkpoint. No frozen kernel was patched; correction
and new GPU qualification are required before claiming arbitrary thresholds.

Sources: frozen Run026 `candidates/k019/kernel.cu` (`rope_gate`), frozen
Run028 `candidates/k049/joint.cu` (`gated`, `gate_pair`), and
`candidates/k050/candidate.py` (`gate_spec`).

### Historical composition and numerical failures

K044's installer selects K033 rather than its intended K036 base. This
previously recorded implementation/hypothesis mismatch is confirmed in source;
the retrospective correctly benchmarks the executed composition. K050 does
not inherit K044. Historical numerical failures remain failures even though
their plot markers are ordinary circles. K017/K018 on ungated A0 remain
unsupported, not assigned made-up timings.

The specialization also assumes immutable weights, fixed shape/device and
non-overlapping use of persistent buffers. It is not a general training,
concurrent-serving, arbitrary-mask, cached-decoding or nonfinite-input API.

## Independent numerical findings and causal limits

The independent least-squares calculation reproduces
`speedup = 0.9531843003 + 3.8586397669 R_model`,
`R_squared = 0.7812571630`, with all 35 checkpoints. Leave-one-checkpoint-out
slopes remain positive (3.629861-5.068416), with R_squared 0.772610-0.823410.
This sensitivity check is descriptive, not an independent-seed confidence
interval. Forty-one checkpoint pairs contradict strict monotonicity.

K050's mean speedup is 1.250205x. The same-fusion no-skip control averages
1.183457x; K050/control averages 1.056401x, helps 19/35, and is 1.311153x
at c30. Disabling attention skipping improves every checkpoint, with mean
speedup 1.267131x. The no-skip control also correlates with R_model
(R_squared 0.496418). Therefore the main scatter alone cannot attribute the
whole trend or gain to sparse multiplication. Fusion, topology, baseline
cost, weights and loss differ across conditions. The ablations establish
a narrower implementation-path contribution, not a pure intervention on
scalar R_model. Comparisons remain normalized ratios from separate processes.

The manuscript's definition credits zero **activation** operands, not zero
weights; the audited canonical projection counter matches this convention.
FP16 canonical R_model and BF16 runtime counts are separate quantities.
"Training-induced sparsity" refers to post-hoc evaluation of the trained
checkpoints, not measuring hardware acceleration during training.

## Figure and manuscript disposition

The current figure remains unchanged at SHA-256
`9f65a78bfe68f96cab04aae9a44349ad460c682584fb91745b44e15ffc39356b`.
Its interpretation requires the caption in Observation05: qualified-only
fixed-checkpoint blue incumbent, numerical failures among gray points,
separate P0 position, all 35 K050 points, percentage ticks with fractional
equation units, and "best" restricted to historical proposals, not ablations.

The user-authorized section is `manuscript/draft/kernel-autoresearch.tex`,
included by `experimental-study.tex`. It reports only the scoped positive
case above and includes the existing run-owned PDF as its main artifact.
`manuscript/draft/kernel-implementation.md` supplies the actual inherited
CUDA snippet, complete execution/qualification contract, code pointers,
ablation table, bug counterexample, source attribution and agent-identity
boundary. No research finding is promoted and no benchmark input changes.

The existing manuscript is a 5.5-inch lightweight reading wrapper, explicitly
not the ICLR submission template. The subsection follows its current section
structure and notation; compilation is not a claim of submission-template fit.

## Tests

- Full bootstrap plus Run029 suite, including new audit tests: **319 passed,
  1 expected Windows/Triton skip**, 24.83s.
- Independently invoked frozen K050 wiring/gate and hybrid-counter tests:
  **8 passed**, 1.99s.
- New counterexample, exhaustive BF16-boundary, independent-fit/timer and
  tampered-gate-summary tests: **4 passed**, 1.06s (included in the full suite).

Test XML is retained under `runtime/audit-final-tests-001.xml`,
`runtime/audit-frozen-components-001.xml` and
`runtime/audit-counterexamples-001.xml`. Publication handoff records their
identities without committing runtime caches or live artifacts. No GPU or
paid model API was launched for this audit.

The final-byte full-suite rerun is `runtime/audit-handoff-tests-001.xml`:
319 passed, 1 skipped, 23.95s. `20_verify_audit_handoff.py` verifies the new
source snapshots, literal CUDA excerpt, resource links, test receipts and
compiled reading copy, retaining `results/audit-publication-001.json`.
All ten pages were rendered at 110dpi and visually checked; the build has
no unresolved references/citations, box warnings or Type 3 fonts. Draft source
snapshots are in `provenance/manuscript-20260908/`; the rest of the draft
retains its existing local-only version-control policy.
