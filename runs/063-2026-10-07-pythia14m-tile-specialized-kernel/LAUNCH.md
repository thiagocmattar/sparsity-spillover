# Run063 launch packet — approval pending

Design approval covers a specialized inference consumer for the two retained
Run060 T2/Ph corner8x16 checkpoints at kappa 0/.1, step712, training seed1234.
Weights, gate semantics, pressure history, validation cache and workload stay
unchanged. No training or gate-selection experiment is included. This packet
requests separate billable launch approval under root `AGENTS.md`.

## Implemented comparison

One physical RTX5090 runs five BF16 B1/T2048 full50304-logit graphs per threshold:
native, generic sparse/no-h-z-skip, and specialized sparse/no-h-z-skip. The
generic CUDA source is byte-identical to Run061. The specialized consumer reads
corner masks directly, uses four projection warps and retains the existing
M8/K16 arithmetic and ordered BF16 branch/residual rounding. It omits short-row
SIMT substitution. Generic no-skip disables tile bypass and short-row SIMT;
specialized no-skip disables tile bypass. All other K050 components, including
the frozen K035 attention kernel, remain shared and unchanged on RTX5090.

Three fresh sequential processes per threshold, 64 fixed inputs and seven
randomized paired passes give 1344 observations per comparison. Synchronized
host latency is primary; equal input staging and setup/compilation are excluded,
while recurring gates/support/projections remain timed. CUDA-event samples,
process ratios, tokens/s, setup and peak memory are retained.

## Numerical failure and qualification scope

The previous local two-block kappa0 smoke failed every optimized path, including
the generic and both no-skip controls. Its loss difference was -.0018486466;
relative L2 passed, but elementwise logit bounds and the .001 loss bound failed.
The original failure artifacts are retained.

`12_trace_numerics.py` supplies identical native hidden inputs to all six layers
and compares components independently. Normalization, QKV/W1, RoPE/V layout and
the generic fused h/z projections/residual match exactly. Only frozen K035
attention differs. `13_attention_backend.py` identifies local native Flash
SDPA and shows that substituting native SDPA only in the shared attention path
makes both consumers match eager logits/loss exactly on the two failing blocks.

Local development qualification uses matched native SDPA on both sides. This
does not qualify the frozen RTX5090 attention path. The scientific executable
has an explicit RTX5090 device check and cannot use that local substitution.
No numerical acceptance bound is relaxed: elementwise atol .25/rtol .02,
relative L2 <= .02, and absolute pooled loss difference <= .001.

Verified local result (`prelaunch/native-cuda-005.json`): all five graphs at
both thresholds match own eager logits and pooled loss exactly on all338
blocks/500 documents per checkpoint. Local native BF16 losses are5.556327955
and5.598824187; these are laptop qualification values, not replacements for
the original quality measurements or RTX5090 losses.296.68s including cold
compilation;4.016GiB peak allocated/4.520GiB reserved. The exact copied-source
inventory and output hashes are in `prelaunch/native-local-receipt-005.json`.
Ten focused tests plus242 bootstrap tests pass together in10.48s. All23
run-local Python sources parse; both shell scripts and the PowerShell guard
pass syntax checks. All1389 immutable input/source identities verify, and all
1422 upload members pass individual byte-count and SHA256 verification.

Mandatory after launch: repeat all 22 CUDA component checks on RTX5090,
precompile the exact frozen chain, then run both threshold smokes (four timing
inputs/eight validation blocks, every graph versus own eager). Stop on any
failure. Every full process must subsequently qualify all five graphs on all
338 blocks/500 documents, 692224 input tokens and 691886 predictions, excluding
and reporting the 1444-token tail. Unqualified evidence is rejected by the
controller and reducer.

The package also fixes a preparation manifest-key mismatch and loads run-owned
diagnostics by explicit path to avoid the import collision documented in Run061.
These fixes change infrastructure, not weights, gates or numerical bounds.

## Hardware, price and duration

Live read on 7 October2026, retained in `prelaunch/catalog-001.json`: Secure
RTX5090, 32GB, LOW availability, USD .99/hour; deep probe confirms LOW stock on
host CUDA13.2. CLI data-center discovery shows LOW RTX5090 stock in EU-RO-1.
The MCP catalog expansion did not return data-center stock; the explicit CLI
read supplies placement evidence. Recheck price/stock immediately before create.

Use one owned Pod named `run063-tile-specialized-kernel-001`, Secure Cloud,
one `NVIDIA GeForce RTX 5090`, EU-RO-1, eligible host CUDA13.2, explicit22/tcp and
registered account SSH key. Image:
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
Pinned application: Python3.12, Torch2.11.0+cu128, Transformers5.12.1,
NumPy2.5.0 and the retained complete package lock; uv0.8.15. Current CLI2.14.0
matches the official latest release inspected this turn.

The laptop supports development qualification, but RTX5090 preserves the
approved hardware comparison and the frozen attention schedule. Previous
Run061 used 4.617–4.623GiB reserved with more checkpoint families; allow
8–10GiB for five graph modes and verify actual fit in the remote smoke.

Initial cloud ETC: **40–75 minutes**, including provisioning, upload, pinned
setup/cold compilation, two smokes, six full processes, diagnostics, verified
download and analysis. Approximate components: provision/upload2–5min;
setup/compile15–30min; smokes2–5min; cohort/diagnostics15–25min;
retrieval/analysis3–10min. Previous Run061 completed in18m21s on a favorable
host; cold compilation and host storage justify this conservative envelope.
Refresh ETC after the actual setup and first smoke.

Maximum: **two GPU hours from provider creation, USD3 total**. At .99/hour,
GPU ceiling is USD1.98. Use40GB container disk plus25GB Pod workspace volume,
no new network volume. Running storage at .10/GB/month is approximately
USD.0178 for65GB over two hours (730-hour month), leaving margin within USD3.
The [official storage page](https://docs.runpod.io/pods/storage/types) confirms
rates and persistence. Price above .99/hour, another GPU, or a larger billable
envelope requires a revised packet.

## Transfer, monitoring and teardown

The sealed upload is `bundles/input-002.tar`:149872640bytes, SHA256
`bae20b46548e1f0cf7acc463b127c18692a929ff27717ae1f0d1bbeed8142f69`.
Its exact per-member inventory and receipt are beside it. It includes both
final inference checkpoint sets, training provenance, complete validation,
frozen dependencies, package pins and run-local scripts. Original checkpoints,
optimizer states and training gradient/OL1 evidence remain retained locally.
This inference run creates no new checkpoint.
Unlaunched bundle001 is retained as a preparation record; bundle002 also
includes the corrected local receipt filename and final transport selection.

The previously confirmed post-hoc inventory is retained: raw paired timings,
full numerical/loss qualification, exact/near-zero counts at0/.001/.01,
per-site/layer RMS/L2, weight norms,8x16 empty tiles, independent MMA/SIMT
counters and standalone generic/corner inspection cost. Logical opportunities
and standalone component timings are not measured DRAM traffic or additive
end-to-end attribution.

All logs, inputs and outputs persist under `/workspace`. Runtime/compiler caches
use fast local container disk; immutable copies are hash-checked before use.
The workload runs detached with persistent events/logs. Arm the identity-scoped
hidden workstation provider-stop guard before upload and the credential-free
remote process-group deadline. Stop compute at two hours while retaining the
workspace for retrieval; terminate only after agreed artifacts verify locally.

Monitor read-only every60seconds, or at the projected completion window if
sooner. Report phase/block/process progress, latest loss/error, throughput,
memory and refreshed ETC. Setup has no loss. Warn/stop on OOM, nonfinite
outputs, failed qualification,900-second leaf timeout, stale scientific progress
over300seconds, disk shortage or approaching deadline. Check cold compiler
activity separately from scientific-progress staleness.

Retrieve every successful/failed attempt, raw timing JSONL/input indices,
numerical/loss evidence, diagnostics/counters, inspection timings, GPU UUID,
source/runtime/setup identities, all logs and SHA256/byte transfer inventory.
Expected output payload is under25MB; actual size and hashes are authoritative.
Verify archive and each member locally, terminate the owned Pod, confirm absence,
disarm its guard and audit remaining resources. Then generate the comparison
table, one PDF and its source-linked observation, and commit accepted evidence.

Current foreign resources: stopped Pod `qa00sm05lmrs7w`,
`run052-sparse-attribution-001`, and pre-existing100GB network volume
`9luykg5yc3`, `sparsity-spillover-shared`. Neither is reused or modified.
No Run063 Pod or other billable resource has been created.
