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

An infrastructure retry may create another attempt under the same unchanged
run. A changed scientific input creates a new numbered run.

## 7. Retrieve, verify, and remove cloud resources

On terminal completion, verify the remote artifact set, build an inventory of
relative paths/bytes/SHA-256 values, transfer the approved files, and verify the
same inventory locally. Only after successful verification may the Pod be
terminated. Re-list Pods and volumes and report any intentional retained cost.

## 8. Report and wait

Report coverage, principal scalar results, artifact locations, failures or
limitations, and cloud cleanup. Do not automatically launch additional analysis
or generate figures/TeX. Wait for the user's analysis instructions.

## 9. Observations and consolidation

Each analysis or figure gets one local observation file with method, coverage,
caption/legend, measured pattern, uncertainty, and nonclaims. Add it to that
folder's index. Only user-approved candidate findings move into
`research/findings/` and the compact `research/INDEX.md`.

If the user requests a manuscript update, change the smallest relevant `.tex`
section and record the exact observation/finding and generating analysis behind
every result-bearing statement or table. The manuscript may also motivate the
next question; it never authorizes that run by itself.

## Mutable planning, immutable evidence

The user may reorder, replace, or cancel planned work without editing a frozen
matrix. What must not be changed in place is evidence: resolved configs,
attempt manifests, raw metrics, checkpoints, observations tied to a figure, and
the approval record for an executed run.
