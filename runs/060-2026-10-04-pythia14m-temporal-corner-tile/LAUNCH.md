# Approved launch

The user explicitly approved this packet: “Approved. Go on. Get back with the
results.” No billable resource or scientific training had started at approval.
User design refinement
selected the inclusive first-corner8x16 gate. This packet covers only the two
14M T2/Ph conditions at 0/.1 in config, with the matched recipe in README.

## Verified preparation and local fit

Focused tests19/19 and bootstrap242/242 passed. CPU smoke passed one boundary
per condition with the real14M architecture, approved random initialization and
exact h-only OL1 capture. Tests cover signed/equality semantics in FP32/FP16/BF16,
tile boundaries, future-token invariance, prefix logits, gradients, gate ports,
checkpoint/recovery serialization, pooled diagnostics and logical counters.

Run058's matched production Ph workload reserved61,412,999,168 bytes (57.20GiB),
72.24% of an80GB H100. The local16GiB laptop and32GB5090 cannot run MB32/T2048
unchanged. Run058 Ph measured five post-warmup complete updates per condition:
medians3.3910/3.3890s, including staging, gradients and optimizer. These are
historical scalar-gate measurements; the new tile operator is not yet calibrated
on CUDA. Both exact-condition preflights will measure five timed full updates,
full validation, full diagnostics, memory and checkpoint roundtrip before training.
Reject skipped/overflow updates, identity drift, >90% reserved memory, failed
diagnostics or median update>6.8s. Do not quietly change batch decomposition.

## Live offer, choice and bounded cost

4 October2026 MCP catalog and current runpodctl GPU listing agree on Secure
catalog rates: H100SXM80GB USD3.49/GPU/hour, H200141GB USD4.59,
5090 USD.99 and B200 USD6.79. Receipts:
[MCP catalog/inventory](prelaunch/catalog-and-inventory.json),
[per-data-center CLI catalog](prelaunch/cli-catalog.json).

Availability differs between the unfiltered MCP capacity reply (H100/H200 High)
and the newer CLI per-center listing (Low). Treat this as placement uncertainty,
not proof that a two-GPU host is allocatable. H100 has eligible centers including
AP-IN-1/AP-IN-2/AP-JP-1, CA-MTL-1, EU-FR-1/EU-NL-1, EUR-IS-3 and US centers.
Recheck stock and the actual creation quote at launch. Do not attach a volume
that unnecessarily constrains placement.

Propose one Secure Pod `run060-temporal-corner-tile-001`, two H100SXM80GB,
one independent condition per GPU. H100 has measured recipe throughput and
memory headroom; H200's extra speed for this workload is unmeasured. If a two-GPU
host cannot allocate promptly, one two-GPU H200 Pod is allowed within
the same total two-GPU/cost ceiling, with explicit execution receipts and the
same runtime/recipe. Do not use additional GPUs concurrently or alter science.

ETC60–90min: preparation/cache15–25min, training40–45min from the historical
calibration, preflight/diagnostics/checkpoints/transfer approximately5–15min,
with overlap and provisioning uncertainty. Refresh from actual tile preflights.
Expected GPU expense aboutUSD7–11 for H100; proposed hard running limit2hours
from provider start. At catalog rates, two H100s for2hours costUSD13.96; two
H200s costUSD18.36. Allow storage/price variation within a maximum incremental
USD30 total. If the live offered total cannot fit this ceiling, do not create it.
Allow at mostUSD.10 estimated storage for the bounded live run; report provider
storage prices/actual creation cost and continuing stopped-volume charges.

Pinned image digest in config,40GB container disk plus50GB Pod volume mounted
at `/workspace`, existing SSH key injected before creation. No new network
volume. Current inventory shows a pre-existing EXITED Run0525090 Pod and the
100GB shared network volume; neither is owned by this launch. The Pod listing
disagrees with the historical Run052 closeout statement of deletion: record it
without deleting/reusing a foreign resource or claiming that the account is empty.

All code/cache/logs/checkpoints are persistent under `/workspace`; the process
is detached and survives terminal loss. A credential-free remote workload guard
and hidden workstation provider-stop guard share the provider-start deadline.
On deadline, stop and preserve unverified artifacts; normal completion transfers
and verifies before terminating the Pod. Stopped storage continues billing until
retrieval/deletion, so complete recovery promptly and report any retention.

## Transfer and post-hoc inventory

Upload the actual committed Git bundle (historically~160MB) and~57MB untrained
random replay, verify SHA256 remotely. Sparse checkout selects source, Run004
dependencies and Run060. Rebuild the pinned MiniPile/tokenizer cache once per
Pod; both canonical train/validation hashes and the realized schedule must match.
Run058 rebuilt/verified the cache in958.9s. Do not upload sixGB over an unmeasured
workstation link.

Retrieve all resolved configs/manifests/events, full validation metrics, all
exact/near-zero and RMS/L2/moment counters, corner/occupancy diagnostics at the
declared cadence, gradient/OL1 boundary metrics, final weight norms/logical
products,24 model checkpoints and6 full recovery states (estimated2.1GB raw,
~1.9GB compressed), runtime/preflight/control logs, and exact byte/SHA inventories.
Retain final checkpoints and immutable cache identities for later measurements.
No clipping frontier or latency work is included. Gradient interaction is collected
during training because it cannot be recovered from weights afterward.

After training, run the scientific verifier remotely, seal terminal artifacts,
copy locally, verify the archive and every member, then verify the full cohort
locally. Produce the requested quality/sparsity/empty-tile tables and PDF figures
with corresponding observation files. Only then terminate owned Pods and confirm
their absence; reconcile any retained resources and provisional/posted billing.

## Monitoring and confirmation

Read-only monitoring every5minutes with bounded idle sleeps: step, finite loss,
tokens/s, refreshed ETC, GPU memory/utilization, disk and actual setup/post-hoc
stage. Check at the projected completion window if sooner. Warn for no new events
for10min, nonfinite loss, any skipped/overflow boundary, >90% memory, disk shortage,
process/coverage/transfer failures or ETC nearing the deadline. No silent rescue
gate/pressure/training variant is permitted.

Both design and launch are now approved, including this post-hoc inventory.
No additional approval is required to execute within this scope and billable
envelope. Record actual provisioning, calibration and teardown below/alongside
the execution receipts.

## Completed execution

Provisioned two H100SXM GPUs at the actual total rate USD6.98/hour. Both production
preflights passed at 3.382s/update and 72.3% peak reserved memory; cache hashes,
initial bytes, schedule and source identity matched. The two scientific conditions
completed 712 updates without overflow/skips. Remote and local scientific
verification passed, and all 144 archived files passed local byte/SHA256 checks.
All24 model checkpoints and 6 recovery states are retained locally.

Analysis041 contains the full comparison, three visually verified PDFs and their
observations. The owned Pod was terminated at approximately 13:51UTC, before the
14:23:51UTC hard deadline. Estimated GPU expense USD10.126-10.215 plus bounded
storage is below the USD30 envelope; posted billing remains partial. No owned
billable resources remain. See `results/compute-closeout.json` and the run README.
