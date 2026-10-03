# Run058 authorized launch packet

The user explicitly approved RunPod execution under the previous run's
principles; no additional confirmation is pending.

Implemented: four fresh T2/Ph/P2 conditions at kappa0/.1, h/z-only unilateral
gates, Ph h-only or P2 h/z OL1, unchanged Run057 recipe, complete validation,
training-time gradient diagnostics, count-pooled all-site zeros/tiles and
logical products, twelve model checkpoints and three recovery states per
condition. Run-local configuration, trainer, diagnostics, scientific verifier,
detached concurrent pipeline, transport and scoped deadline guards are ready.

Verification:19 focused tests pass, including exact h/z placement, no a/m
gate/pressure, twelve versus six pressure tensors, pressure mean,
threshold/equality/backward mathematics, histogram/tile count pooling,
checkpoint/recovery serialization, full-block coverage and exact schedule.
Four real14M CPU T16/MB1 FP32 mechanics smokes pass with the declared replay
initialization; these do not establish CUDA workload fit. Full bootstrap:
242 tests passed. Initial adaptation checks exposed T4-only hook/pressure
assumptions; corrected before launch, with no scientific training started.

Local workload: production MB32/GAS32 T2048 pressure historically reserves
approximately57GiB; the user's16GiB laptop cannot fit it unchanged. Run057's
eight H100 exact-workload preflights measured3.386--3.609s/update and72.24--72.84%
reserved memory. The new four-condition preflight will measure actual T2 fit,
five complete timed updates after warmup, full validation, checkpoint roundtrip
and named diagnostics. Reject overflow/skips, excessive memory, invalid
identities or median update above6.8s before training.

Live catalog/capacity queried3 October2026: H100SXM80GB, four-GPU Secure
capacity Medium with CUDA13.0/13.2 drivers available, aggregateUSD13.96/hour.
Pinned CUDA12.8 runtime is driver-compatible. H200 availability Low,
USD18.36/hour for four. Choose H100's verified workload and moderate capacity
over H200's unmeasured advantage, avoiding provisioning/calibration uncertainty.
B200's higher catalog rate has no measured T2 advantage for this small model.

Create one Secure Pod named run058-t2-p2-versus-ph-001, four H10080GB,
one condition per GPU, globally eligible placement, pinned image digest from
config, 40GB container disk plus50GB Pod volume at/workspace, existing SSH key.
Do not attach or modify the pre-existing shared network volume or foreign Pod.
Logs, checkout, verified cache and checkpoint artifacts persist in the Pod
volume and all work is detached from the SSH terminal.

Estimated end-to-end ETC60--90minutes: preparation/cache~15--25min,
training~40--45min, preflight/validation/diagnostics/checkpoints/transfer
~10--20min with overlap/uncertainty. Expected compute~USD14--21; maximum3hours
at live rateUSD41.88 plus storage belowUSD0.10, bounded byUSD45 total.
Credential-free remote workload guard and hidden workstation provider-stop
guard use the same deadline from provider startedAt. Stop preserves unverified
artifacts; normal completion uses verified transfer then deletion.

Source bundle~160MB and~57MB random replay are hash-checked after upload.
Sparse checkout retrieves only source and required Run004/Run058 code.
The pinned MiniPile/tokenizer builder runs once and must reproduce both
canonical cache hashes before the worker barrier; no six-GB workstation cache
upload is needed. Source commit, runtime, configuration, replay and realized
training schedule identities are verified.

Retrieve all manifests, resolved configs, every boundary event, complete
scalar/counter metrics, activation/weight/gradient/logical diagnostics,48 model
checkpoints and12 full recovery states (~4.1GB payload,~3.8GB compressed),
actual preflight/environment/control records and signed byte/hash inventories.
Verify archive and every member locally, run the scientific verifier, preserve
the exact sealed artifact bytes, then delete the owned Pod and confirm absence.
Record estimated and lagged posted billing distinctly.

Monitor read-only at five-minute intervals with bounded sleeps, reporting
current step/loss, throughput and refreshed ETC. During active setup report
the actual preparation stage. Warn for stale events10min, nonfinite loss,
overflow/skipped boundaries, excessive reserved memory, missing files, disk
pressure, process failure or ETC approaching the deadline. A shorter projected
completion window replaces the ordinary interval.
