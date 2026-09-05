# Run 025: agentic sparse-kernel specialization for Pythia

Status: **pilot concluded with a P0 full-model correctness blocker; no GPUs
remaining; optimization search not started**.
Execution source baseline: `afad780`. See the append-only
[calibration observation](observations/001-calibration-gates.md) and
[cost/ETC closeout](CALIBRATION_CLOSEOUT.md).
Created 2026-09-05 at the user's explicit request to reserve a new run folder
and write a plan. This documentation-only reservation does not bypass design
confirmation for implementation or the repository's launch review.

## Decision in brief

Revised 2026-09-05 following the user's **$40 total budget** and explicit
instruction to **start from Sakana's kernel**. This revision replaces the
earlier larger-budget proposal; that proposal remains in Git history only.

Start from the official Sakana repository at pinned commit
`661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`, not a new kernel written from
scratch and not solely the historical raw-ELL adapter. Preserve separate
upstream, minimal Pythia-adaptation, and agent-optimized source identities.
Audit/use the optimized TwELL path wherever the exact Pythia contract permits;
document necessary correctness adaptations before measuring search gains.

Recommend one RTX 5090 32 GB for development, with a short H100 SXM 80 GB
check. At this budget, its lower hourly rate buys more iteration time than
RTX PRO. Batch-1 inference should fit with headroom when loading one model at
a time, but calibration must verify actual workspaces. RTX PRO is a shorter
search fallback if memory or availability prevents using RTX 5090.

The proposed study covers all 36 selected trained checkpoints: 12 each at
14M, 70M, and 410M. Develop on 18 endpoint sentinels; freeze the implementation
before testing the 18 interior-kappa checkpoints. Primary workload is full
sequence prefill at batch 1, length 2,048. Batch 32 is an optional sentinel
check only after the core evidence is secured, not a promised full matrix.
Both FFN and attention remain in scope, with separate contribution tests.

| Envelope | GPU allocation | Compute estimate | Total cash ceiling |
| --- | --- | ---: | ---: |
| Calibration gate, included below | Up to 2 RTX 5090 hours + 1 H100 hour | $4.07 | $5 |
| Recommended complete case study | Up to 32 RTX 5090 hours + 4 H100 hours | $32.84 | **$40** |
| Conditional RTX PRO fallback, not additional | Up to 16 RTX PRO hours + 3 H100 hours, less prior spend | $35.11 before prior spend/storage | **Same $40** |

Refreshed community quotes: RTX 5090 $0.69/hour, RTX PRO $1.69/hour, H100 SXM
$2.69/hour. The recommended allocation leaves $7.16 for storage, billed
transfer/retries, billing uncertainty, and any incremental agent charges.
No separately billed agent API is planned: use the existing authorized agent
session. If that route incurs incremental charges, they must fit inside $40
by reducing search allocation before launch; they are not excluded extras.

These are **bounded search budgets, not measured completion forecasts or a
promise of speedup**. Allow approximately 1-3 calendar days after local
implementation, availability, and agent-session continuity. Initial local
preparation is estimated at 6-12 working hours. Calibration determines actual
candidate throughput and final-evaluation ETC. Search stops early to protect
validation, transfer, and teardown budgets.

## Paper-level evidence under this ceiling

Keep strong dense and minimally adapted Sakana baselines, one fully logged
agentic trajectory, all 36 primary checkpoint results, untouched interior
thresholds during tuning, component ablations, repeat timings, complete
validation, and a fixed-kernel H100 check on 18 sentinels.

Remove the three paired agent/no-feedback replicates and the H100 retuning
search. The result can be a reproducible **agent-assisted systems case study**.
It cannot establish general agent superiority or the causal advantage of
feedback over another search strategy. A positive full-model gain and a
positive relation to `R_model` remain hypotheses, not completion requirements
that justify spending past the cap.

## Implemented calibration package

The confirmed design now has a run-local Sakana-derived P0 adapter, rotating
full-model timer, fixed numerical/full-validation gates, input hashes for all
36 checkpoints, phase-specific transfer bundles, and bounded worker/trial
records. **242 bootstrap tests pass**, including 23 focused Run 025 tests.
The retained 14M four-family CPU smoke also passes. None of this substitutes
for compiling and testing the CUDA kernel on the deployment GPU.

See [launch review and exact test scope](LAUNCH_REVIEW.md),
[source/correctness audit](SOURCE_AUDIT.md), and
[local smoke evidence](prelaunch/local-smoke.json).

P0 covers the four linear sites; QK/PV attention remains dense. Sparse
attention, the stronger compiled/graph dense selection, search scoring, and
the frozen all-36 final evaluator remain subsequent approved-design work.
No optimized winner is claimed. The approved sequential **$5 calibration gate**
concluded at approximately **$1.38 all-in** (billing provisional). Both GPUs
passed 48 primitive cases under compute-sanitizer. A0-14M passed full validation;
A1-H-14M failed its fixed elementwise logit gate, stopping the eight-model loop.
The separate H100 upstream backbone control measured 1.1654x. These unmatched
workloads do not establish an architecture/hardware effect or validate P0.

## Documents

- [Argument chain and falsification criteria](ARGUMENT_CHAIN.md)
- [Scientific design and step-by-step experiment](EXPERIMENT_PLAN.md)
- [Bounded agent-loop contract](PROGRAM.md)
- [GPU, ETC, costs, transfer, and monitoring playbook](COMPUTE_PLAN.md)
- [Refreshed $40 planning snapshot](planning/gpu-market-snapshot-budget40.json)
- [Original historical market snapshot](planning/gpu-market-snapshot.json)

## Workflow / progress

- [x] Read operational definitions, prior runs, and relevant manuscript sections.
- [x] Audit the previous benchmark and the scope of upstream kernel claims.
- [x] Query live GPU catalog and existing resources without creating anything.
- [x] Write the argument, proposed controls, search budget, and stop conditions.
- [x] Revise for the user's $40 ceiling and explicit official-Sakana starting point.
- [x] Human confirms/refines the scientific design and diagnostic inventory.
- [x] Implement calibration evaluator, P0 linear adapter, and bounded trial records.
- [x] Pin checkpoint/cache identities, proposed runtime and budget; record controller defaults and attribution limits.
- [x] Pass focused tests and the complete bootstrap suite; present calibration launch definition.
- [x] Obtain explicit calibration launch approval and verify live lease/guard readiness.
- [x] Execute bounded calibration; record partial coverage and correctness failure.
- [x] Report calibration decision: repair/qualify P0 before optimization; larger-size ETC remains unmeasured.
- [ ] Seal CUDA-verified P0, strongest dense configuration, search evaluator and controller provenance before K001.
- [ ] Execute one Sakana-derived closed loop, at most 40 candidates / 20 GPU-hours.
- [ ] Freeze winner; verify all 36 checkpoints, component effects, and H100 transfer.
- [x] Retrieve/hash-verify pilot artifacts; terminate both Pods; confirm zero Pods/endpoints.
- [x] Write pilot observation and cost/ETC closeout, including failed attempts.
- [ ] Execute final-study retrieval and publication figures if the full study proceeds.

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
a result of this task. Reserve the existing volume's prorated charge during
execution too; do not mistake the $40 allowance for GPU-only funds.

## Local evidence used for this plan

- [Run 022: 14M narrow-width failure and positive control](../022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/README.md)
- [Run 023: 70M implementation and sentinel results](../023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/README.md)
- [Run 024: 410M implementation and sentinel results](../024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/README.md)
- [Analysis 014: matched H100 measurements and regressions](../../analyses/014-2026-09-04-pythia70m-vs-410m-sparse-kernel-sentinels/README.md)
- [Operational methods](../../research/METHODS.md), [metrics](../../research/METRICS.md), and [compute contract](../../research/COMPUTE.md)
