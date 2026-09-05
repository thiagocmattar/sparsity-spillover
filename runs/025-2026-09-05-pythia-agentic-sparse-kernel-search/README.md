# Run 025: agentic sparse-kernel specialization for Pythia

Status: **planning only; design pending; no experiment code or GPU launched**.
Created 2026-09-05 at the user's explicit request to reserve a new run folder
and write a plan. This documentation-only reservation does not bypass design
confirmation for implementation or the repository's launch review.

## Decision in brief

Use one RTX PRO 6000 Blackwell Server Edition 96 GB for development, followed
by a short H100 SXM 80 GB transfer test. The choice is provisional until a
small calibration measures valid candidate evaluations per dollar. Optimize
exact inference on retained checkpoints, not training or the sparsity gates.

The proposed study covers all 36 selected trained checkpoints: 12 each at
14M, 70M, and 410M. Develop on 18 endpoint sentinels; freeze the implementation
before testing the 18 interior-kappa checkpoints. Primary workload is full
sequence prefill at batch 1, length 2,048; batch 32 is secondary. Both FFN and
attention are in scope, with explicit tests of their separate contributions.

| Envelope | GPU allocation | Compute estimate | RunPod budget including reserve |
| --- | --- | ---: | ---: |
| Calibration gate, included in full budget | Up to 4 RTX PRO hours + 2 H100 hours | $12.14 | $15 |
| Recommended controlled study | 36-48 RTX PRO hours + 6-8 H100 hours | $76.98-$102.64 | $85-$115 |
| Smaller exploratory alternative | 18-24 RTX PRO hours + 4 H100 hours | $41.18-$51.32 | $45-$60 |

Prices are the 2026-09-05 live community-cloud catalog quotes: RTX PRO
$1.69/hour and H100 SXM $2.69/hour, not booked prices. The smaller alternative
omits the replicated, budget-matched agent-feedback comparison and cannot
support that claim. It is an alternative, not an additional phase.

These are **bounded search budgets, not measured completion forecasts or a
promise of speedup**. Allow approximately 2-4 calendar days for the recommended
study after implementation and capacity availability. Initial implementation
is estimated at 6-12 working hours, subject to the attention/harness audit.
Agent-model/API charges are not included; its execution route and any separate
token budget must be pinned before launch. No external paid agent API is
provisioned by this plan.

## Documents

- [Argument chain and falsification criteria](ARGUMENT_CHAIN.md)
- [Scientific design and step-by-step experiment](EXPERIMENT_PLAN.md)
- [Bounded agent-loop contract](PROGRAM.md)
- [GPU, ETC, costs, transfer, and monitoring playbook](COMPUTE_PLAN.md)
- [Live planning snapshot](planning/gpu-market-snapshot.json)

## Workflow / progress

- [x] Read operational definitions, prior runs, and relevant manuscript sections.
- [x] Audit the previous benchmark and the scope of upstream kernel claims.
- [x] Query live GPU catalog and existing resources without creating anything.
- [x] Write the argument, proposed controls, search budget, and stop conditions.
- [ ] Human confirms/refines the scientific design and diagnostic inventory.
- [ ] Implement run-local evaluator, kernel adapter, and bounded trial logging.
- [ ] Pin model/controller identity, environment, checkpoint/cache hashes, and budgets.
- [ ] Pass focused tests and the complete bootstrap suite; present launch definition.
- [ ] Calibrate memory, compilation, correctness, timing, transfer, and GPU cost/ETC.
- [ ] Report calibration decision: proceed, revise with the human, or stop.
- [ ] Execute three paired closed-loop/no-performance-feedback search replicates.
- [ ] Freeze winner; verify all 36 checkpoints and test H100 transfer/retuning.
- [ ] Retrieve and hash-verify artifacts; terminate Pods; reconcile billable resources.
- [ ] Write observations and PDF figures; report failures as well as improvements.

## Evidence boundary

The current negative result concerns a particular exact raw-ELL adaptation.
It does not establish a universal limit of Sakana, TEAL, or unstructured
sparsity. A successful new run would demonstrate a bounded engineering case
study, not that agentic coding is universally superior. A higher `R_model`
remains a logical opportunity, not a promised runtime gain.

No manuscript TeX is changed by this planning task. Relevant paper targets are
the runtime results and elementwise-versus-executable-sparsity discussion in
`manuscript/draft/sections/06_results.tex` and `07_discussion.tex`.

At the planning resource check, RunPod had zero Pods and zero endpoints. The
existing 100 GB network volume was retained unchanged; no GPU is spending as
a result of this task. Its ongoing storage charge is separate from this run.

## Local evidence used for this plan

- [Run 022: 14M narrow-width failure and positive control](../022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/README.md)
- [Run 023: 70M implementation and sentinel results](../023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/README.md)
- [Run 024: 410M implementation and sentinel results](../024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/README.md)
- [Analysis 014: matched H100 measurements and regressions](../../analyses/014-2026-09-04-pythia70m-vs-410m-sparse-kernel-sentinels/README.md)
- [Operational methods](../../research/METHODS.md), [metrics](../../research/METRICS.md), and [compute contract](../../research/COMPUTE.md)
