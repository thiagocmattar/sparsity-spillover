# Run 029: matched retrospective Pythia-14M kernel evaluation

Status: executing GPU preflight for the user-confirmed design. Design and GPU launches are
explicitly pre-approved within USD20 and 16 cumulative GPU hours. No training,
new optimized kernel, manuscript edit, or new paid agent API is authorized.

## Question and fixed contract

Re-evaluate historical kernel implementations under a single RTX5090 protocol
to reconstruct agent-guided specialization progress. Compare the final policy
and an explicitly labeled Sakana-derived Pythia adapter across all35 existing
checkpoints, and separate skip-path and fusion contributions.

Only implementation varies. Retained random-pretraining checkpoints use seed1234,
step712 and their original weights, gates, thresholds, pressure interventions,
and initialization provenance. No optimizer or backward pass is run. The primary
workload is BF16, batch1, T2048, uncached causal inference with all50304 logits.
The same native dense PyTorch-SDPA CUDA-graph reference is paired with each
candidate. An unmodified eager stock model anchors numerical checks; the shared
Run028 scaffold avoids a Transformers graph-capture mask-dispatch confound.

All qualified measurements require all338 complete blocks from all500 MiniPile
validation documents (692224 input tokens,691886 prediction tokens,1444-token
excluded tail). Timing uses64 fixed validation identities,7 paired passes and3
fresh processes. Geometric means pool paired ratios, not ratios of medians.
Graph setup, compilation and equal input staging are reported but untimed;
all recurring input-dependent packing, zero detection and full logits are timed.

The retrospective cohort is fixed at c01(A0),c11(A4/kappa0),c25(A7/kappa.5),
c30(A7+OL1/kappa.5). Its progress line is the best fully qualified c30 result,
initialized by the dense implementation at1x; qualification is checkpoint-local,
not a claim that an early candidate works on the whole cohort. All other outcomes
remain visible in the data and supplementary figure. Final comparisons cover
all35 checkpoints, including failures and slowdowns. Canonical FP16 R_model
counts remain separate from BF16 executed-work diagnostics.

## Candidate preservation and eligibility

`archive/root/` preserves original repository-relative source layout so inherited
imports resolve exclusively to frozen run-local copies. `provenance/archive.json`
records every byte/hash/origin. This avoids editing historical kernel files or
silently importing changing code from earlier runs. `provenance/candidates.json`
indexes K001-K050, P0, original settings, exclusions and the paper-iteration map.
Eligible14M implementations retain their algorithm and settings; K011's twelve
original masks are distinct configurations at one historical proposal. Generic
14M-capable early implementations can acquire their first full-model timings in
this retrospective study, explicitly labeled as such. Other-size-only policies
and standalone attention primitives are archived but not ported into new14M
algorithms. At most64 configurations are budgeted.

The unmodified Sakana source at661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5 is a
separate reference: Hopper producer and restricted non-gated output shapes do
not provide an unchanged Pythia14M/RTX5090 full-model comparator. P0 is its
previously documented signed-exact shape-adapted descendant, not the published
fused FFN and not an optimized new Blackwell port. Failed P0 qualification is a
valid possible outcome. No H100/SparseLM ratio is mixed into this curve.

## Interpretation and diagnostics

Support requires qualified full-model improvement under the common protocol;
positive skip-toggle benefit supports sparse-path contribution. Null/negative
effects, a stronger early candidate, or poor R_model association refute the
corresponding narrow claims and will be retained. Neither candidate numbering
nor a cumulative maximum proves independent trials, monotonic per-candidate
improvement, causal agent superiority, or a universal sparsity speed law.
This addresses the paper's runtime-realization argument; no TeX is changed.

Retain raw timing pairs, all block-level numerical checks and pooled loss,
source/config/environment hashes, failure examples, canonical integer counts,
and final untimed exact/near-zero counts,RMS,weight norms,executed skip counters
and row-NNZ histograms. Gradient conflict is unavailable in inference. Existing
checkpoints and precise cache identities are retained; no full logits dump is
needed beyond compact failure examples.

## Compute, monitoring and closeout

Planning ETC:6-12 local implementation hours,7-14 billable GPU hours plus local
analysis. Historical Run028105 nine-mode processes took4517.7s and peaked at
4.324GiB allocated; a new calibration will supersede this estimate. One physical
RTX5090 is used, with no concurrent scientific workload. GPU rate must be at most
USD0.99/hour, maximum16 aggregate hours, total budgetUSD20 including retries and
storage. No new network volume is planned; existing100GB EUR-IS-1 storage stays
unchanged. Code, inputs and outputs live under persistent `/workspace`.

Monitor detached jobs every120s (shorter near projected completion), reporting
progress, loss/error, throughput, memory and refreshed ETC. Investigate nonfinite
outputs, CUDA errors, stale events or projected envelope overruns. Reserve time
for transfer; incrementally copy completed artifacts and verify SHA256 locally
before normal Pod teardown. A separate deadline guard is the backstop.
Publication outputs are PDFs with observation files and provenance. Commit only
code, manifests, observations and intended publication artifacts, never weights,
caches, credentials or files still being written. Do not push.

## Execution record

At2026-09-07T19:38:33Z, created Community Pod`l22xsah56cwbl1`,
`run029-rtx5090-001`, at a returned GPU rateUSD0.69/hour. Physical GPU UUID:
`GPU-d77f736c-1ebe-6277-380f-b544e7271074`; RTX5090,32607MiB, driver570.195.03.
Image digest is pinned in script08. Container20GB + Pod volume50GB;
no network volume was created or attached. The existing100GB volume is untouched.
Independent hidden guardPID38872 was armed and verified, with stop deadline
2026-09-08T11:38:33Z. Controller reserves15 minutes before that for retrieval.
The stop backstop retains storage; normal verified closeout deletes this Pod.

Local full bootstrap plus run checks:305 passed,1 Windows-only Triton skip
(23.96s). Linux run checks:61 passed (10.82s), including all54 eligible
installation interfaces. The three reduction tests additionally verify raw
pair coverage, estimated-intercept regression, checkpoint-fixed selection,
and rejection of invalid/unqualified incumbents. GPU smoke tests use only
4 timing inputs,2 passes,8 validation blocks and are never paper observations.

Archive preparation froze1374 files and validated88 distinct historical
candidate execution hashes. Catalog:50 original K proposals,8 excluded
other-size/standalone implementations,42 eligible proposals,53 eligible K
configurations plus P0. K011's12 masks share one proposal ordinal. The fixed
matrix has648 retrospective and525 final fresh processes,1173 total, each
with its own matched dense denominator. Final ablation candidates are measured
in separate randomized fresh-process pairs against native; a ratio between
their normalized speedups is not a within-process direct toggle measurement.

The1972899840-byte input tar matched local and remote SHA256
`155d9ba5de8f22aa89cdf46738a8c4b4bb6033a6af8b2da723b89ed9b6fb0c84`.
The corrected code003 package matched
`5e8f2627dae8bdc8a21f0d950176be23fd9c4673450ffa93912bf55a38ea6cc5`;
the subsequent catalog ordering correction is sealed in each phase plan.
All source snapshots, checkpoint files, validation data and canonical
provenance records subsequently passed remote byte/hash verification.

Infrastructure preflight fixes, with failed logs retained: Windows extended
paths were required for vendored CUDA headers and deep provenance records;
manifest-driven packaging replaced silent deep-path omissions. Ubuntu's
externally managed Python rejected system pip, so uv is installed inside
a workspace bootstrap venv. Setup002 completed package installation before
input upload, then its input check correctly rejected the not-yet-extracted
data. Verification passed after complete transfer. A catalog test detected
platform-dependent filename ordering; explicit case-insensitive ordering
now matches Windows/Linux without changing source bytes or eligibility.
No kernel algorithm, gate, checkpoint or numerical bound was modified.

Incremental output receipt`results/retrieval-smoke001.json` verifies the first
153 files (1438008 bytes), including20 completed smoke evaluations. Every
retrieval checks the tar SHA256, exact member set, and each file SHA256/size.
Only completed leaf artifacts are admitted; live controller/log snapshots
are copied to a separate immutable retrieval record before local-view refresh.

## Reproduction entry points

The self-contained source layout is `archive/root`; replay never imports live
historical run folders. Retain `inputs/` or the verified input bundle above:
weights and data are intentionally not committed. `provenance/inputs.json`
identifies every original and run-local checkpoint file, validation stream,
and canonical logical-product record. Install the pinned environment with
script05, run script04`verify`, and execute script07 with phase`smoke`,
`calibration`, then`scientific` and an explicit authorized deadline.

Script03 owns one exclusive phase lock and one child GPU process at a time.
Existing completed leaves are not rerun or selected by speed; interrupted
leaves receive new attempt names. It rejects changed phase sources on resume,
stops on reference failure/CUDA context faults, and retains other candidate
failures. After complete retrieval, script10 reduces the matrix and script12
creates the three vector PDFs; script13 verifies and imports retrieved output.
Launch control is deliberately separate: a reproduction is not permission
to provision or spend, and script08 must receive a fresh live price/resource audit.
