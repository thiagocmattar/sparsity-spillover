# NNN — Experiment title

**State:** proposed | design-approved | implemented | launch-approved | running |
completed | failed | consolidated

## Request and detailed interpretation

Record the user's plain-language request and the agent's precise interpretation.

## Goal / hypothesis

- Expect:
- Supports if:
- Refutes if:

## Manuscript relationship

- Manuscript goal, term, equation, table, or proposed claim:
- Operational definition used here:
- Known difference from the current draft:
- Possible observation, figure, table, or TeX consequence:

## Interpretation choices

List every choice that could have produced a scientifically different run.

## Design

### Varies

### Matched controls

### Model and initialization

### Data, order, batch, budget, optimizer, and schedule

### Full validation

State 500 documents, 338 complete sequences, 692,224 input tokens, and the
1,444-token excluded tail unless this approved run deliberately changes them.

### Topology, gate, and pressure

Keep these as separate fields.

### Seeds and uncertainty

### Diagnostics and post-hoc retention

Record which metrics run during training, which diagnostics run before cloud
teardown, and whether the final checkpoint is retained/transferred.

### Interpretation limits

## Approval record

- Design approved by user: TODO (date and concise scope)
- Launch approved by user: TODO (date and exact local/cloud envelope)

## Implementation

Numbered scripts, reusable modules changed, and why each exists.

## Verification

Commands, test scope, results, and limitations.

## ETC and execution plan

- Calibration evidence:
- Local fit/ETC:
- Proposed location:
- RunPod GPU/count/rate/max duration/max cost, if applicable:
- Storage and transfer inventory:
- Monitoring cadence and warning conditions:
- Teardown plan:

## Attempts

Each attempt gets an immutable row with status, command, path, environment,
failure/retry rationale, and artifact inventory.

## Results

Numbers and links to observations. Do not write a conclusion before inspecting
coverage and sanity checks.

## Candidate findings

Not filed until the user approves consolidation.

## Where we stopped

Exact next action for a fresh session.
