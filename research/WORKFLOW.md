# Human-Guided Experiment Workflow

This is a conversation and record-keeping protocol, not an executable approval
engine. The research plan may change whenever the user chooses; an executed run
remains an immutable record of what actually happened.

## 1. Plain-language request

The user states the experiment or diagnostic in ordinary language. The agent
uses `DEFINITIONS.md` and, for paper-connected work, the living manuscript. It
asks only about choices that materially change the scientific question.

The acknowledged manuscript/operational crosswalk in `MANUSCRIPT.md` is
accepted background. Operational contracts execute by default, so those known
differences are not repeated during every design. Reopen one only when the user
requests a manuscript-specific variant or it creates a material ambiguity for
the proposed experiment.

No real numbered folder is created yet.

## 2. Detailed interpretation and design approval

The agent returns a concrete proposal covering:

- question, hypothesis, varying conditions, and matched controls;
- model, initialization, data order, global batch, budget, optimizer, schedule,
  seeds, and checkpoints;
- topology, nonlinear gate, pressure method, and pressure sites as separate
  choices;
- all-500-document validation coverage;
- training-time diagnostics and post-hoc analyses;
- manuscript question/definition/claim affected and the operational method
  that will execute, including any new or design-material difference;
- evidence that would support or refute the hypothesis and important nonclaims.

The user refines or explicitly approves that design. This approval authorizes
implementation only.

## 3. Create one numbered record

Find the largest existing run number, allocate the next three-digit number, and
create `runs/NNN-YYYY-MM-DD-description/` from the template. Record the request,
interpretation, approval scope, and unresolved `TODO`s before code.

Keep run-specific files here. A typical folder grows only as needed:

```text
runs/NNN-YYYY-MM-DD-description/
  README.md
  config.yaml
  01_prepare.py
  02_train.py
  03_diagnose.py
  04_plot.py
  observations/INDEX.md
  observations/O001-*.md
  figures/*.pdf
  artifacts/attempts/NNN-*/...
```

Numbers express execution/dependency order. Do not create empty scripts or
directories just to match the example.

## 4. Implement and verify

Implement only the approved experiment. Keep novel behavior local. Move a
primitive into `src/` only when a second run needs the same scientific behavior.

Run focused mathematical and placement tests, then the complete bootstrap test
suite. Perform a production-shaped smoke/calibration that uses the proposed
model, sequence length, precision, microbatch, accumulation, pressure path, and
diagnostics. A small smoke run tests mechanics; it is not paper evidence.

Before handoff, confirm every new test is visible to Git with `git status` and,
if needed, `git check-ignore -v`. Never ignore the repository `tests/` tree:
Runs 009 and 010 demonstrated that ignored tests can pass locally while being
silently absent from the commit that future agents receive.

Keep every pytest temporary base under the single repository-root
`.pytest_tmp/` directory. The default test configuration uses
`.pytest_tmp/default`; when a separately named base is useful, use a child such
as `--basetemp=.pytest_tmp/focused` or `--basetemp=.pytest_tmp/full`. Do not
create sibling `.pytest_tmp_*`/`pytest_tmp*` directories or pytest temp
directories inside `runs/`, `analyses/`, or other workflow folders. These
directories are disposable test scratch, never experiment evidence.

## 5. Launch packet and launch approval

Before any scientific execution or cloud creation, report:

- implementation and exact tests passed;
- measured memory, step time, validation time, throughput, and uncertainty;
- local fit/impact and local ETC;
- proposed local or cloud location and why;
- for RunPod, live GPU price/capacity, maximum hours and total cost, image,
  storage, transfer inventory, and teardown guard;
- monitoring cadence and stale/nonfinite/resource warning conditions;
- measurements that must be collected before teardown and checkpoints retained.

The user refines or explicitly approves this exact envelope. Material changes to
the envelope require renewed approval.

## 6. Execute and monitor

Create a new immutable attempt, snapshot the resolved config, and publish a
`running` manifest before work. Write progress to `events.jsonl`. A long process
must survive loss of the control connection.

Monitor read-only with bounded sleeps (`tools/watch_run.py` is the minimal local
helper). Report actual step/tokens, finite losses, diagnostics, throughput,
memory/disk, and refreshed ETC. Do not equate a live process or a `Running` Pod
with a healthy scientific run.

Readers of live logs and JSONL files must not take a blocking file lock. Run 008
showed that PowerShell `Get-Content` can block on an actively written event file
and leave monitor-only processes consuming memory. Prefer a short Python reader
that opens, reads, and exits; if an auxiliary reader hangs, identify and remove
only that reader, never the authoritative cohort process.

When the refreshed ETC is shorter than the normal monitoring interval, wait
until the projected completion window and check once. Poll sooner only for an
already declared warning. The wait itself is not a reason to query an unchanged
process repeatedly.

An infrastructure retry may create another attempt under the same unchanged
run. A changed scientific input creates a new numbered run.

Classify the boundary explicitly. A failed environment install, unreachable
host, or unusably slow filesystem before the scientific manifest/attempt begins
is an infrastructure failure. Preserve its log and identity, then retry the
unchanged approved definition inside the same run. Once scientific inputs or an
attempt change, allocate the appropriate immutable attempt or a new run.

## 7. Retrieve, verify, and remove cloud resources

On terminal completion, verify the remote artifact set, build an inventory of
relative paths/bytes/SHA-256 values, transfer the approved files, and verify the
same inventory locally. Only after successful verification may the Pod be
terminated. Re-list Pods and volumes and report any intentional retained cost.

For a distributed run, distinguish per-attempt verification from global cohort
verification. Each worker must be independently acceptable before its Pod is
removed; a verifier that requires all conditions cannot be used as that worker's
only teardown gate. After all verified attempts are co-located locally, run the
global comparator/order/coverage verifier once and retain its output.

## 8. Report and wait

Report coverage, principal scalar results, artifact locations, failures or
limitations, and cloud cleanup. Do not automatically launch additional analysis
or generate figures/TeX. Wait for the user's analysis instructions.

Provider billing is asynchronous evidence. Record the query timestamp, scope,
window, posted total, and whether the latest bucket covers teardown. Label an
incomplete snapshot provisional and refresh it later; never present a changing
posted bucket as a settled invoice.

## 9. Observations and consolidation

Each analysis or figure gets one local observation file with method, coverage,
caption/legend, measured pattern, uncertainty, and nonclaims. Add it to that
folder's index. Only user-approved candidate findings move into
`research/findings/` and the compact `research/INDEX.md`.

At run closeout, update the run README, terminal verification artifact, launch
or deployment record, and compact `research/INDEX.md` in the same coherent
change. Search for stale states such as `running`, `unapproved`, and `pending`
before committing; Run 009 showed that a correct run record can otherwise
coexist with a stale top-level index.

If the user requests a manuscript update, change the smallest relevant `.tex`
section and record the exact observation/finding and generating analysis behind
every result-bearing statement or table. The manuscript may also motivate the
next question; it never authorizes that run by itself.

## Mutable planning, immutable evidence

The user may reorder, replace, or cancel planned work without editing a frozen
matrix. What must not be changed in place is evidence: resolved configs,
attempt manifests, raw metrics, checkpoints, observations tied to a figure, and
the approval record for an executed run.

## Version-control closeout

Commit source, configs, tests, compact JSON/JSONL evidence, diagnostics,
inventories, and verification records. Exclude credentials, caches, datasets,
model weights/checkpoints, compressed transfer archives, PIDs, active logs, and
pytest scratch. A checkpoint inventory and content hash may be committed even
when the checkpoint bytes remain local and ignored.

Before staging, enumerate untracked non-ignored files and inspect the largest
ones. After staging, review `git diff --cached --stat`, `git diff --cached
--check`, and the staged path list; run a credential-pattern scan over the
staged content. Commit only after the scientific verifiers and relevant tests
pass. Push only when the user explicitly requests it, and confirm that the
remote branch resolves to the new commit.
