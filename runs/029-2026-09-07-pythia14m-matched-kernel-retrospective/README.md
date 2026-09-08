# Run 029: matched retrospective Pythia-14M kernel evaluation

Status: completed and verified. All1173 scheduled process outcomes,35 final
diagnostic passes and three revised paper PDFs are retained locally. The Pod
is deleted. Design and launches were pre-approved within USD20 and16 cumulative
GPU hours; no training, new optimized kernel, manuscript edit or paid agent API
was added. Current assets and interpretation are in the closeout section below.

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

For a fresh reproduction, create a new human-approved run record using these
frozen sources and input identities; do not overwrite this executed artifact
tree or regenerate its provenance. Existing phase plans intentionally resume
by skipping terminal leaves rather than silently starting a new experiment.
Windows Git operations on the vendored tree require `git -c core.longpaths=true`.
Run-local attributes preserve snapshot/JSON bytes; script14 independently
compares every staged Git blob with its working-file bytes and rejects weights,
datasets, caches, credentials directories and live run artifacts.

### Preflight complete; matched matrix started

All60 GPU smoke jobs completed at20:19:42Z:54 numerical passes,6 failures
(K020-K024 andK026 atc30), and no compilation/execution failure. Both P0 and
K050 passed the four cohort smoke checks. Short8-block results are not used
as paper measurements. Receipt`retrieval-smoke002.json` verifies443 files,
3598664 bytes. The source/protocol foundation is committed as`6d843dc`;
1518 run files (32234883 bytes) were independently verified against staged
Git blob identities, excluding weights/data/caches/live evidence.

All six calibration processes then passed all338 validation blocks. P0's
three c30 paired speedups are0.8180133,0.8177230,0.8179256; K050's are
1.7825312,1.7832513,1.7827064. Native and K050 BF16 pooled loss is
5.8313006630; P0 differs by approximately-2.823e-9. These are runtime BF16 checks, not the retained FP16 diagnostic
loss. Calibration took95.2517s wall time including process lifecycle.
Individual evaluation time was13.61-14.05s without diagnostics, and21.76s
for K050's full diagnostic pass. Peak allocated memory was3.034GiB.
Receipt`retrieval-calibration001.json` verifies490 files,6235778 bytes.

The1173-job scientific matrix launched detached asPID14175 at approximately
2026-09-07T20:25:06Z on the same physical GPU. Refreshed ETC is4.5-6 hours
plus retrieval/verification, roughlyUSD3.5-5 total Pod spend including setup
and storage, belowUSD20 and the unchanged16-hour ceiling. Monitoring uses
the completed-process wall estimate, including interpreter lifecycle, rather
than the narrower child-only timer. No additional kernel search or tuning
occurs in this retrospective matrix.

`results/calibration-summary-001.json` retains the exact calibration scalars,
full lifecycle estimate (5.173h before candidate/topology adjustment), and
runtime identity. `provenance/pip-freeze.txt` is a byte-identical copy of the
verified Linux dependency lock, including all observed transitive versions;
fresh reproductions should use it alongside the pinned CUDA12.8 image and
PyTorch CUDA wheel index, not rely on future resolution of unpinned dependencies.

During execution the user requested longer idle sleeps to reduce unnecessary
status checks and token use. From approximately20:49Z, operational monitoring
uses600-second PowerShell sleeps between checks, with milestone/error updates.
This supersedes the initially planned120-second observation cadence only;
the unchanged controller still enforces each600-second leaf timeout and the
independent16-hour stop guard remains armed. No scientific input is changed.

Incremental backups verify every inventoried byte but are not transactional
controller checkpoints. In scientific008, the live controller index advanced
to1021 entries after leaf enumeration had selected1020 completed results; the
one newer result was therefore not in that tar. This was detected by a local
read-only cross-reference check, not a benchmark failure. Final analysis and
teardown require a quiescent final export and strict reconciliation of all1173
planned result references, timing pairs, numerical checks and35 diagnostics.
No result is inferred from an intermediate controller index.

## Completed evaluation and current paper assets

The scientific controller finished at2026-09-08T01:17:05Z after17498.2703s
(4h51m38s), averaging241.33 complete process lifecycles/hour. Outcomes:
855 qualified,312 numerical failures,6 explicitly unsupported, with no
compilation/execution failures. All1167 timed processes completed338-block
qualification. Reduction reconciles391 three-process comparisons and522816
raw native/candidate timing pairs on the same64 validation identities and one
physical GPU. The six unsupported outcomes are K017/K018 on ungated c01.

The matched c30 incumbent improves at paper proposals9,10,11 and42:
1.057046x,1.097161x,1.602386x and1.783029x. Native1x is a real pre-search
reference, not a renormalized first kernel. The42 eligible proposals retain
all53 historical configurations; the independently benchmarked P0 is separate.

K050 passes all35 final checkpoints, with equal-checkpoint geometric mean
1.250205x and range1.005938-1.783175x. Unweighted estimated-intercept OLS gives
`S = 0.9531843 + 3.8586398 R_model`, `R_squared = 0.7812572`, with canonical
R_model expressed as a fraction. This supports a conditional association,
not a universal or exact runtime law.

Sparse-path contribution is not uniform: K050/no-skip averages1.056401,
helping19/35 checkpoints and hurting16/35; at c30 it is1.311153. The fused
no-skip control already averages1.183457x over native. Disabling attention
skipping is faster on all35 checkpoints (mean control speedup1.267131x versus
K0501.250205x), so attention skip counts must not be presented as net speed
benefit. Full35-checkpoint BF16 activation, weight, logical and executed-work
diagnostics are retained separately from canonical FP16 R_model.

P0, the existing Sakana-derived Pythia adapter, qualifies on5/35 checkpoints
and averages0.867570x on those five. The other30 fail elementwise logit gates
despite finite outputs and passing relative-L2/loss checks. No qualified
all35 P0 average or unchanged SparseLM/H100 comparison is claimed.

Current publication assets (single panels, grids, no bars or variant legend):

- [Matched progress](figures/01-matched-autoresearch-progress-r02.pdf)
  and [Observation01](observations/01-matched-autoresearch-progress.md).
- [Individual candidate outcomes](figures/02-matched-individual-candidates-r02.pdf)
  and [Observation02](observations/02-matched-individual-candidates.md).
- [Canonical R_model versus acceleration](figures/03-matched-rmodel-speedup-r02.pdf)
  and [Observation03](observations/03-matched-rmodel-speedup.md).

Revision1 PDFs are superseded proofs retained for audit. Revision2 moves a
reference annotation off the early steps, displays quarter-step tick values
exactly and keeps the complete OLS line in view. Measurements, regression,
point selection and progress definition are unchanged. All three current PDFs
were rendered and visually inspected; fonts are embedded and graphics vector.

### Final retrieval, verification and teardown

The quiescent final bundle contains9035 files/505698600 inventoried bytes;
its523530240-byte tar has SHA256
`5ac93fe118ec053014cc15f43a833cbf26d3c9febd84e90da918e3123470d81b`.
Receipt`results/retrieval-scientific-final001.json` verifies exact tar members,
file sizes and SHA256. Strict script10 reconciled all1173 planned result
references, raw timing/quality coverage and all35 diagnostics before teardown,
resolving the partial live-index mismatch. Frozen inputs and sources were also
reverified remotely and locally. All checkpoints, validation inputs, raw
artifacts, failures, dependency lock and verified bundles remain local.

The owned Pod was deleted successfully (HTTP204) by2026-09-08T01:22:31Z,
after verification, and its identified local deadline guard was stopped.
The post-delete audit finds zero Pods and endpoints. Only the pre-existing
100GB shared network volume remains, unchanged. Creation-to-deletion allocation
was at most5.732746h. GPU plus Pod-disk estimate is at mostUSD4.02 versus the
USD20 envelope; posted billing remains lagged (USD3.3506 at closeout), not a
final invoice. `results/cloud-closeout-001.json` records the live API responses,
price, timing, estimate and storage boundary. No push or manuscript edit occurs.

The final local bootstrap/run suite passed312 tests with one expected
Windows/Triton skip (25.51s). Publication verification confirms three single-page,
vector-only PDFs with embedded subset Unicode TrueType fonts. Results are
recorded in `results/verification-001.json`; Linux preflight had61 passing checks.
To reproduce local reporting from retained raw evidence: run scripts10,16,
then12 with a fresh publication revision in the new reproduction record; existing
result paths are historical records, not permission to overwrite them or relaunch
GPUs. Script17 uses the available Poppler commands and verifies
current data/input/source hashes, test evidence and the publication PDFs.
