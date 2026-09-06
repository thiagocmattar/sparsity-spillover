# Run 025: agentic sparse-kernel specialization for Pythia

Status: **The bounded search, frozen Blackwell matrix, fresh-process
replication, fixed-policy H100 transfer, component probes, and direct
fixed-`R_model` confirmations are complete. Sparse QK/PV attention and the
strongest compiled-dense comparator remain outside the achieved scope. No GPU
is running.**
Execution source baseline: `afad780`. See the append-only
[calibration observation](observations/001-calibration-gates.md) and
[cost/ETC closeout](CALIBRATION_CLOSEOUT.md).
Created 2026-09-05 at the user's explicit request to reserve a new run folder
and write a plan. This documentation-only reservation does not bypass design
confirmation for implementation or the repository's launch review.

## Interim Blackwell completion addendum (2026-09-06)

The user subsequently approved the bounded continuation and all required
launches within the same $40 ceiling. The continuation retained P0 and tried
K001 through K016. It froze separate policies for each architecture: K013 for
14M, K016 for 70M, and K010 for 410M. On one NVIDIA RTX PRO 4500 Blackwell it
then evaluated all 36 checkpoints at batch 1 and sequence length 2,048 with 80
paired timing samples each and complete 338-block numerical/quality validation.
Thirty-two final deployments pass; four fail and remain part of the record.

[Analysis 015](../../analyses/015-2026-09-06-pythia-agentic-kernel-search/README.md)
owns the count-reconciled table, qualified regressions, fixed-`R_model`
implementation comparisons, and two inspected publication PDFs. The qualified
within-size OLS results are `R2=0.235` at 14M, `0.003` at 70M, and `0.611` at
410M. Those heterogeneous results support implementation specificity, not a
universal runtime mapping.

The evidence archive was retrieved and matches its stable remote post-package
SHA-256. The only self-inventory mismatch is a declared 46-byte controller-log
race; all 4,100 stable files match. The final Pod was deleted, and a fresh
control-plane check reports zero Pods and zero endpoints. The existing 100 GB
standard volume remains intentionally retained in EUR-IS-1.

At this interim point the experiment was still **partial against its
preregistered minimum paper package**. QK/PV matmuls remained dense SDPA; the
strongest compiled-dense comparison, three fresh-process repetitions,
contribution ablations, and fixed H100 transfer had not yet been completed.
The later addendum below records the completed replication and transfer work.
Final timing continued to use the first 16 fixed seed-2500 training-cache
blocks rather than the planned seed-2504 validation sample. Complete
validation coverage is unaffected.

At the post-teardown billing snapshot, the account window covering this work
reports $7.1028 in Pod GPU charges, $0.0648 in Pod disk, and $0.2236 in standard
storage ($7.3913 total). This is an account-window total and therefore a
conservative Run 025 debit, not invented per-Pod attribution. The account
balance was $34.8766 and ongoing spend was $0.01/hour from retained storage.
Charging the whole window to Run 025 leaves $32.6087 of the $40 ceiling. Live
community H100 quotes were $2.59/hour for NVL and $2.69/hour for SXM, but no
H100 was launched by this addendum.

## Replication and hardware closeout addendum (2026-09-06)

The approved continuation added three independent eager Python processes for
each of the 36 Blackwell checkpoint deployments, the 18 preregistered endpoint
sentinels on an H100 NVL without retuning, 40 component probes across the two
GPUs, and a direct same-checkpoint/same-`R_model` experiment on the RTX PRO 4500.
Every primary fresh process and every component probe ran complete 338-block
validation. The four known frozen-policy numerical failures repeated in all
three Blackwell processes and remain excluded from qualified fits.

Fresh-process regressions remain conditional rather than universal. On RTX PRO
4500, within-size qualified OLS `R2` is 0.347 for 14M, 0.0027 for 70M, and 0.636
for 410M. On H100 NVL the corresponding six-sentinel values are 0.321, 0.051,
and 0.700. The H100 410M slope orders six implementations that are all slower
than native, so association and useful acceleration are separate outcomes.
Every architecture nevertheless has at least one qualified native speedup on
at least one GPU: 1.0725x for 14M and 1.0452x for 70M on H100, and 1.0192x for
410M on RTX.

Attempt `rtxpro4500-004` then held the trained checkpoint, workload, and
canonical integer `R_model` counts fixed while changing only implementation
policy. It ran 36 prespecified P0/K013, K009/K016, or K004/K010 processes and 12
adaptive P0/K001 14M confirmation processes. All 48 passed the complete
validation gate. The primary optimized-over-baseline medians across six fresh
process pairs per architecture are 1.0152x at 14M, 1.0113x at 70M, and 1.0064x
at 410M; 16 of 18 process pairs favor the optimized implementation. The final
14M robustness policy K013 is a deliberate negative contrast: it is only
0.9370x and 0.9477x as fast as P0 on the two tested endpoints. This supports
implementation sensitivity and tuning risk at fixed logical opportunity.

The component probes do not establish sparse QK/PV attention. They show that
eligible attention-projection linears exceed native break-even in all 18
probes (1.0034x--1.0254x), whereas FFN-only eligibility exceeds break-even in
11 of 22 probes (0.8497x--1.0246x). The paths are non-additive, consistent with
packing, dispatch, and fragmented-execution overhead, but direct low-level
counters were not serialized.

The complete attempt-004 archive is 731,849 bytes with SHA-256
`0753655f73ccd2bd8586265c1558641949fc4b6b112684570b891f3b00a815d8`.
Both phase verifiers pass locally after transfer. The Pod and its deletion
guard were removed; authoritative control-plane queries report zero Pods and
zero endpoints. The pre-existing 100 GB standard network volume remains
intentionally retained. The final account-wide 2026-09-05--06 billing snapshot
is recorded in
[`billing-closeout.json`](autoresearch/launch-control/rtxpro4500-004/billing-closeout.json):
$18.2629 total, including $17.8409 Pod GPU, $0.1303 Pod disk, and $0.2917
standard storage. It is a conservative account-window debit, not fabricated
per-Pod attribution, and leaves $21.7371 of the $40 ceiling before delayed
charges and continuing retained-volume storage.

Closeout verification passes 35 focused analysis/controller tests and the
complete 242-test repository bootstrap suite.

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

## Implemented calibration package (historical snapshot)

The confirmed design now has a run-local Sakana-derived P0 adapter, rotating
full-model timer, fixed numerical/full-validation gates, input hashes for all
36 checkpoints, phase-specific transfer bundles, and bounded worker/trial
records. **242 bootstrap tests pass**, including 23 focused Run 025 tests.
The retained 14M four-family CPU smoke also passes. None of this substitutes
for compiling and testing the CUDA kernel on the deployment GPU.

See [launch review and exact test scope](LAUNCH_REVIEW.md),
[source/correctness audit](SOURCE_AUDIT.md), and
[local smoke evidence](prelaunch/local-smoke.json).

P0 covers the four linear sites; QK/PV attention remains dense. At this
calibration stage, sparse attention, the stronger compiled/graph dense
selection, search scoring, and the frozen all-36 final evaluator remained
subsequent approved-design work, and no optimized winner was yet claimed. The
approved sequential **$5 calibration gate**
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
- [x] Seal CUDA-verified P0, the evaluator, and controller provenance before K001.
- [ ] Seal the strongest compiled-dense comparator (native eager remained the final reference).
- [x] Execute one Sakana-derived closed loop: P0 plus K001--K016, below 40 candidates and the $40 ceiling.
- [x] Freeze one policy per size and evaluate all 36 checkpoints on Blackwell.
- [x] Repeat final timing in three fresh processes on the development GPU.
- [x] Complete the prespecified FFN/attention-projection contribution tests; QK/PV remains dense.
- [x] Transfer the frozen policies to the 18 H100 endpoint sentinels without retuning.
- [x] Retrieve/hash-verify pilot artifacts; terminate both Pods; confirm zero Pods/endpoints.
- [x] Write pilot observation and cost/ETC closeout, including failed attempts.
- [x] Retrieve and hash-verify the Blackwell evidence; delete the Pod and recheck the control plane.
- [x] Produce the Blackwell result table, fixed-`R_model` comparisons, and publication figures.
- [x] Close the achieved paper package with an explicit partial-study label for dense QK/PV and the missing compiled-dense comparator.

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
