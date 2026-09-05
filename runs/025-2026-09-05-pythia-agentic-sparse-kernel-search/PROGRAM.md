# Proposed agent-loop contract

This is a design document, not an instruction to start a search now. It is
inspired by a fixed-evaluator autoresearch pattern, not an imported training
program. The goal is correct full-model acceleration, not a favorable plot.

## Responsibilities and mutable surface

Use a small run-local evaluator and candidate directory, not a generalized
research platform. The evaluator owns inputs, reference execution, numerical
gates, timer boundaries, metric aggregation, budgets, and final selection.
The coding agent proposes kernel/adapter changes and explains the expected
mechanism. An evaluator result, not an agent-written claim, determines whether
a trial passes. Seal evaluator hashes before candidate 1 and verify them
before/after each trial. Review the sealed evaluator separately from candidate
code; hash checking alone is not a security sandbox or proof of unbiased tests.

Editable: run-local CUDA/Triton/C++ kernels, shape/GPU dispatch configuration,
legal adapter integration, static weight transforms, and reusable workspace
management. All changes must stay inside the approved mathematical function.

Read-only to candidates: checkpoints, cache identity and token lists, evaluator,
references, complete validation targets, metrics, quality thresholds, timer
implementation, and source-run records. Candidates receive training-split
development examples, not an interface to tune on final-checkpoint scores.
Infrastructure credentials remain outside candidate processes. Candidate
workers do not create Pods, access billing, or need arbitrary network access.

Disallowed: input/output memorization; checkpoint-ID dispatch; dropping work
from the timer; returning stale/asynchronous outputs; changing precision below
the contract; row-cap truncation; modifying gates, kappa, masks, weights, or
loss evaluation; benchmarking a weakened dense reference; editing historical
runs; and hiding failed trials. Dense fallback is allowed only when logged and
charged inside the same full-model timer.

## One closed-loop iteration

1. Read the latest development measurements and profile summary. State one
   bottleneck hypothesis, expected gain, and correctness risk.
2. Create a numbered immutable candidate from the incumbent. Record source
   hash, parent ID, change rationale, proposed shape/device coverage, agent
   identity/settings, tool actions, token use, and wall time.
3. Build in a fresh worker; run focused mathematical, memory, and adapter tests.
   A failed build/test is a recorded attempt, not a missing data point.
4. If correct, run paired primitive and full-model development measurements.
   Report packing/dispatch overhead and true sparse execution, not only FLOPS.
5. Apply the sealed score and promotion rule. Confirm a candidate gain in fresh
   timing blocks. Keep the incumbent if correctness or improvement fails.
6. Append results and synchronize candidate source/log artifacts locally.
   State the next hypothesis. Never rewrite a past candidate or its outcome.
7. Stop at the first time/candidate/token/cost limit. Retain the best verified
   candidate, even if it is the starting baseline or all-dense fallback.

Starting candidate limits are 30 attempts and 6 GPU-hours per trajectory,
including model-response waits while the GPU is rented, compilation, failed
trials, and evaluation. Proposed worker limits: 12 minutes per build and
20 minutes per whole candidate; calibration must verify these against the
actual workload before sealing them. A timed-out experiment is unsuccessful,
not evidence of a slow steady-state kernel. Detect hung CUDA workers with a
separate process watchdog rather than trusting the candidate to terminate.

Three consecutive infrastructure failures stop the trajectory for diagnosis;
they do not justify unlimited retries. Two reproducible numerical failures
from the same approach require a new hypothesis before another attempt.
Resource exhaustion is recorded with memory/shape and a tested fallback,
not silently solved by dropping the largest model or batch.

## Matched no-performance-feedback arm

Use the same agent, code starting point, tools, correctness checks, and upper
budgets. Generate and seal the performance-change proposals before exposing
intermediate candidate timing/profile results to the agent; the initial
baseline/profile summary is identical in both arms. Only correctness/build feedback is
available for repair. Repairs consume attempts/time/tokens. The evaluator
benchmarks valid proposals and selects the highest development score after
the arm finishes. Keep the arm's budget and data accounting visible.

Alternate/randomize arm order within each pair on the same GPU, with no
concurrent benchmarks. Record actual resource use: equal upper budgets do
not guarantee equal token usage or number of valid trials. Compare best-so-far
scores at common elapsed GPU-cost points and report unused budget.

The paired IDs are 2501/2502/2503; use isolated fresh contexts/checkouts and
sealed results between pairs. Their stochasticity concerns the search, not
independent model-training seeds. Three pairs support descriptive evidence,
not a strong population-level statistical claim.

## Agent execution prerequisite

Before launch, pin the actual coding model/version and reasoning setting,
prompt/context policy, tool visibility, model-call and input/output-token caps,
and accounting route. Prefer the existing authorized agent environment; do
not silently provision a separately billed external API. If a reproducible
continuous proposer cannot be run in that environment, resolve the execution
route and priced cap before renting a long-lived GPU.

The existing interactive session can prototype a propose/test iteration, but
is not by itself evidence that six isolated, uninterrupted research trajectories
are operational. Smoke-test the controller's stop/resume and log persistence
without paid GPU time. Do not use an unbounded "never stop" prompt or rely on
a live SSH terminal to keep a trial alive.

## Required trial record

Retain candidate/parent hashes, arm/replicate, timestamp, agent configuration,
prompt/response/tool provenance with secrets excluded, token counts, build
environment and logs, tests and errors, measured shapes/inputs, paired latency
samples, full-model score, quality deltas, profile summary, sparse coverage,
allocated GPU time/cost, decision, and stop reason. Static tuning sweeps and
new kernel algorithms must have distinct labels.

Final evaluation occurs only after freezing. If it fails, report the failure
and a previously frozen valid fallback; do not quietly turn the final set into
another development loop. A follow-up scientific design gets a new run.
