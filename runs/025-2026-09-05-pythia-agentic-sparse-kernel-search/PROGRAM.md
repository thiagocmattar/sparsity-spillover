# Proposed agent-loop contract

This is a design document, not an instruction to start a search now. It is
inspired by a fixed-evaluator autoresearch pattern, not an imported training
program. The goal is correct full-model acceleration, not a favorable plot.

Current design: one trajectory under a $40 total cash ceiling. Start from
official Sakana U0, seal the minimal correct Pythia adaptation P0, then record
optimization descendants K001 onward. The larger replicated comparison is
superseded and is not part of this program.

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

Starting limits are 40 attempts and 20 RTX-5090 GPU-hours in one trajectory
(at most 8 search hours under the shorter RTX-PRO fallback),
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

## Evidence without a separate agent comparator

Use trajectory ID 2501. Publish its complete history rather than a selected
success story. Rebuild P0 and the final candidate, repeat paired timings in
fresh processes, and run frozen component/change ablations. Search trials are
not independent training seeds or replicated agent experiments.

No no-feedback arm, agent-model comparison, or H100 retuning is funded.
State the case-study limitation prominently. Repeated timing validates the
artifact's performance; it does not establish the probability that an agent
will independently rediscover it.

## Agent execution prerequisite

Before launch, pin the actual coding model/version and reasoning setting,
prompt/context policy, tool visibility, model-call and input/output-token caps,
and accounting route. Use the existing authorized agent environment with no
new paid API provisioning. Do not assume its incremental charge is zero:
confirm accounting before launch. Any metered charges must reduce the GPU
allocation inside the same $40 total, not be billed as excluded extras.

The existing interactive session can prototype a propose/test iteration, but
is not by itself evidence that a persistent search worker survives disconnection.
Smoke-test the controller's stop/resume and log persistence
without paid GPU time. Do not use an unbounded "never stop" prompt or rely on
a live SSH terminal to keep a trial alive.

## Required trial record

Retain candidate/parent hashes, trajectory ID, timestamp, agent configuration,
prompt/response/tool provenance with secrets excluded, token counts, build
environment and logs, tests and errors, measured shapes/inputs, paired latency
samples, full-model score, quality deltas, profile summary, sparse coverage,
allocated GPU time/cost, decision, and stop reason. Static tuning sweeps and
new kernel algorithms must have distinct labels.

Final evaluation occurs only after freezing. If it fails, report the failure
and a previously frozen valid fallback; do not quietly turn the final set into
another development loop. A follow-up scientific design gets a new run.
