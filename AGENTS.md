# Agent Instructions

## Purpose and authority

This is a guided research repository for activation sparsity in transformer
pretraining. The human chooses what to investigate. The agent clarifies,
implements, verifies, runs, monitors, retrieves, and consolidates. Do not invent
new research direction or silently select among scientifically different
interpretations.

The research plan is a mutable human guide. It is not an executable permission
system. Explicit user confirmation, not a plan parser, authorizes a design or a
launch.

The manuscript is shipped in this repository under `manuscript/` and is a
living scientific direction. It may sharpen terminology, motivate work, and
evolve from approved findings, but it is not evidence or a launch plan. Do not
force textual alignment: surface scientifically meaningful differences between
manuscript, operational definitions, and implementation.

## Read first

1. `research/INDEX.md`
2. `research/DEFINITIONS.md`
3. `research/DATA.md`
4. `research/MANUSCRIPT.md`, `manuscript/README.md`, and the relevant TeX file
   under `manuscript/` for paper-connected work
5. The active run or analysis `README.md`, if one exists
6. `research/METHODS.md` and `research/METRICS.md` before changing scientific code
7. `research/COMPUTE.md` before estimating or launching work

If operational research documents disagree with files or artifacts, report the
discrepancy. The manuscript/operational differences already acknowledged in
`research/MANUSCRIPT.md` are accepted background: use the operational contracts
by default and do not repeat those differences during every design. Surface a
manuscript difference only when the user requests the manuscript variant or a
new difference would materially change the proposed experiment.

## Two confirmations before execution

### 1. Design confirmation

For a plain-language experiment request, first state in detail:

- the question or hypothesis;
- what will vary and what will remain matched;
- model, initialization, data, complete validation coverage, seeds, budget,
  optimizer, activation topology, gate, pressure method/sites, and checkpoints;
- diagnostics and post-hoc measurements to collect;
- manuscript goal, term, equation, table, or proposed claim affected, and the
  operational definition that will execute; mention only new or
  design-material manuscript differences;
- what would support and what would refute the hypothesis;
- remaining ambiguities and interpretation limits.

Ask the user to confirm or refine this interpretation. Do not create the real
numbered run folder or write experiment code before confirmation.

### 2. Launch confirmation

After design approval, create the next
`runs/NNN-YYYY-MM-DD-description/` folder and implement only that run. Then
report:

- files and behavior implemented;
- exact test and smoke scope with results;
- local resource fit and estimated ETC;
- proposed local or RunPod execution definition;
- live cloud price, maximum duration/cost, storage, transfer, and teardown plan
  when cloud is proposed;
- artifact and checkpoint transfer inventory;
- monitoring interval and warning conditions.

Ask for explicit launch approval. Design approval is not launch approval.

## Run ownership

Everything specific to one experiment stays in its run folder: config, numbered
scripts, README, observations, local artifacts, and PDF figures. Cross-run work
belongs in a numbered `analyses/` folder. Do not grow a central plotting module.

A run folder is an append-only record after execution. Fixing a scientific input
creates a new numbered run. Infrastructure retries get a new attempt directory
inside the same unchanged run and are explained in its README.

## Pre-launch post-hoc checklist

Before launch, explicitly ask which measurements may be needed later. At minimum
consider activation exact/near-zero counts, activation RMS/L2 statistics,
per-layer weight norms, gradient conflict/OL1 boundary metrics, logical-product
counters, clipping thresholds/sites, and whether the final checkpoint must be
retained. Gradient interaction cannot be reconstructed from a checkpoint alone.

If uncertainty remains, retain the final checkpoint and enough cache identity to
run the diagnostic later. Do not terminate cloud compute until the agreed
artifacts have been copied and verified locally.

## Scientific invariants

- Pythia pretraining uses random initialization from the pinned architecture
  config. Released weights imply continuation/fine-tuning and must be named.
- Site aliases and hook placement are exact; read `research/METHODS.md`.
- `l1_naive` and `orthogonal_l1` are different interventions.
- Pressure sites, gate sites, gate operator, and threshold are independent fields.
- Validation uses all 500 MiniPile validation documents by default. Evaluate all
  338 complete 2,048-token blocks and report the 1,444-token excluded tail.
- Pool integer counts before dividing. Do not average percentages across batches
  or layers unless that is the stated estimand.
- Exact zero, near-zero mass, logical opportunity, and measured speedup are
  different quantities.
- `R_block` and `R_model` are logical-product opportunities, not runtime gains.
- `R_model_max` is the manuscript-led analytic all-zero reach ceiling for the
  declared topology, architecture, and full-sequence workload. It is not an
  observed zero rate or speedup; store its integer counts and explicit unit.

## Simplicity

- Write the minimum code that answers the approved question.
- Do not build policy engines, catalog parsers, schedulers, generalized workflow
  frameworks, or broad validators for hypothetical future runs.
- A validator checks types, ranges, and scientific invariants; it does not
  hard-code one experiment's learning rate, cadence, or batch decomposition.
- Prefer focused modules with one responsibility. When a file becomes difficult
  to review in one sitting, split by scientific responsibility, not by arbitrary
  line-count compliance.
- Shared code moves to `src/` only after a second run needs it. Existing bootstrap
  code is limited to methodology already reused in the source study.
- Tests prioritize mathematical behavior, hook placement, aggregation, data
  coverage, and artifact serialization.
- Do not build a manuscript generation framework. Add the smallest `.tex`
  section or generated table needed by an approved paper task and retain its
  evidence provenance.

## Execution and monitoring

- Do locally when the representative workload fits with headroom and the user
  accepts its ETC and impact on the local machine.
- Use RunPod when local memory is insufficient or the local ETC is unacceptable
  and the user approves the live billable envelope.
- For RunPod work, load the installed `runpod` router skill first. Use current MCP
  schemas or `runpodctl --help`; do not copy stale CLI syntax or prices.
- Long work must write logs and artifacts to persistent storage and survive a
  disconnected terminal. Monitor read-only with bounded sleep/poll intervals.
- During expected idle work, wait instead of busy-polling. For Windows local
  runs, use PowerShell `Start-Sleep` for the agreed monitoring interval, then
  perform one read-only status check; use the platform-equivalent idle wait
  elsewhere. Do not repeatedly query an unchanged process between intervals.
- Every monitoring update reports progress, current loss, throughput, and a
  refreshed ETC. If the refreshed ETC is shorter than the normal interval,
  sleep until the projected completion window and check then; check sooner only
  for a previously declared warning condition.
- Never expose credentials. Discover existing resources before creation.
- Verify transferred artifact hashes, terminate the Pod, and confirm no unintended
  billable resources remain.

## Figures and observations

- Save publication figures as PDF only, beside the run or analysis that created
  them.
- Every figure has a corresponding observation Markdown file containing the
  question, method, coverage, legend/caption, result, caveats, and source script.
- Per-folder `observations/INDEX.md` summarizes its observation files.
- Consolidate user-approved findings upward into `research/findings/` and keep
  `research/INDEX.md` to roughly one screen.
- Update or add manuscript TeX only when the user requests it or approves the
  finding/narrative change. Result-bearing TeX must link to its source
  observation and generating analysis when applicable.

## Verification

Run focused tests while implementing and the full bootstrap suite before launch.
Do not claim completion from a green test that does not cover the requirement.
Reconcile README claims, configs, artifacts, and actual files before handoff.

## Version-control closeout

- After completing and verifying a coherent change, review the scoped diff,
  stage only the intended files, and commit it before handoff unless the user
  explicitly asks not to commit or the repository is unsafe to commit.
- Before committing, exclude credentials, datasets, model weights, caches,
  temporary files, and unintended generated artifacts. Never commit files that
  are still being written by an active run.
- Use a commit message that accurately describes the state captured. Do not
  fabricate historical commits, dates, or intermediate states that Git did not
  record.
- Report the commit hash at handoff. Do not push unless the user explicitly
  requests it.
