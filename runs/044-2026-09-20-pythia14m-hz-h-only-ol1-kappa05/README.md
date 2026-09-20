# Run044: missing Pythia-14M T2/Ph endpoint at kappa=0.5

The user requested an exact repeat of Run041 at kappa=0.5 on RunPod, completing
its 0/.01/.05/.1/.5 threshold grid. That explicit configuration selects the
scientific design below. The user subsequently confirmed the full Run041
measurement package and explicitly approved the USD10 launch envelope.
Status: complete. Training and all three K050 processes passed verification;
all agreed artifacts are local and both Run044 Pods are deleted.

Final validation loss is **5.536255**. Pooled h/z exact-zero fractions are
**99.841%/99.841%** and model-wide logical-product opportunity is **5.339%**.
K050 takes **0.506323 ms** versus **0.718438 ms** for native CUDA graphs:
**1.4189x** paired geometric-mean speedup on RTX5090, BF16, B1/T2048.
See the [result and interpretation limits](observations/001-final-results.md).
Estimated GPU cost is **USD4.286**, below the approved USD10 cap; provider
billing is still incomplete and this estimate excludes storage.

## Question and matched contract

Measure how the missing high-threshold T2/Ph endpoint changes validation loss,
activation zeros, model-wide logical sparsity and final K050 inference latency.
Only kappa changes from Run041. There is exactly one condition,
`hz-h-ol1-kappa-0p5`. The operational topology is `HZ`, meaning h and z only;
it is independent of the bootstrap's unrelated historical `A2` identifier.
Both gates are one-sided: retain x>=0.5, otherwise zero; equality and the
surviving identity gradient are preserved. The h gate replaces GELU, and z
acts immediately before attention W_o. OL1 targets only six post-gate h
tensors, averaging tensor means equally, with lambda=1 and trust budget=1.

Pinned random Pythia-14M architecture revision
7386d9a4ae45aef494a6e704910394def3037fc5: six layers, width128, FFN512, four
heads. Replay only Run041's retained random step-zero parameter bytes, never
trained or released weights. Parameter hash
ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57;
realized schedule hash
f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa.
Model/data seeds1234; pinned MiniPile/tokenizer revisions and both complete
token-cache hashes follow config.yaml and research/DATA.md. All1,000,000
training documents; EOS per document. 712 updates, MB32 x accumulation32,
global1024, sequence2048, 1,493,172,224 input tokens, wrapping714 blocks.

Unchanged fused AdamW (.9,.95), eps1e-8, WD.1 except bias/LayerNorm;
task-gradient clipping1; LR.001 to .0001, 1% warmup and pre-step cosine.
Dynamic FP16, FP32 parameters/moments, flash SDPA, no dropout or activation
checkpointing; no skipped update is acceptable. Validate after step1 and
from the final checkpoint: all500 documents,338 complete blocks,692224 input
tokens,1444 excluded tail. Final activation and eager logical passes use the
same complete coverage.

Retain all eight diagnostic sites' per-layer integer exact/near-zero counts
at0/.001/.01, moments/RMS/L2, named parameter norms, all six logical operation
counts, R_block/R_model, and the analytic HZ ceiling
1,006,632,960 / 18,825,609,216 products per B1,T2048 full forward
(0.053471467958893806). Every boundary records gradient conflict, OL1
projection/trust geometry, clipping and loss scale. Model checkpoints:
0,1,2,4,8,16,32,64,128,256,512,712; optimizer/scaler/RNG:256,512,712.
The final checkpoint and exact cache identity are retained. The post-hoc
question has been presented explicitly; the inherited package has no clipping
sweep. Additions require a stated change before launch.

This completes the manuscript's T2/Ph placement grid under the existing
operational contract. Higher qualified sparsity or lower measured latency at
an acceptable loss supports a useful high-threshold endpoint; degradation,
slowdown or failed numerical qualification limits that interpretation.
This is one seed and budget, with no claim of universal superiority, causal
isolation of h versus z, or runtime speedup inferred from logical counts.
No manuscript edit or finding promotion is included.

## Execution proposal

One Secure H100 SXM80GB, one independent training worker, no distributed
gradients; current live quote USD3.49/hour, HIGH availability on20September.
Pinned Run041 image digest in config.yaml;40GB container,50GB persistent
/workspace, broad data-center placement. Run041's exact workload reserved
61,412,999,168 of85,017,493,504 bytes and trained in2,426-2,430 seconds.
It does not fit the local12GB GPU; no local CUDA training is proposed.
Expected end-to-end completion1-2h including installation/cache, exact
preflight, training, diagnostics, retrieval and final latency. Actual ETC is
refreshed from the new preflight. H100 is the measured fast option for this
14M recipe; H200 has not been measured for this workload.

Training Pod stop deadline2h from creation. A later isolated Secure RTX5090
at USD0.99/hour has a90min maximum, with setup overlapped only near training
completion. Total GPU envelope USD8.465; total incremental cap USD10 includes
temporary storage/transfer allowance. No new network volume. Existing Run042,
Run043 and the shared volume are separate resources. Provider stop preserves
unretrieved Pod storage, which would continue billing until recovery/deletion.

Before scientific training, run six real-data boundaries (one warmup, five
timed), full validation, both activation/logical diagnostics and checkpoint
serialization. Verify runtime, random initialization, realized data order,
six h pressure tensors, no overflow/skipped boundary, >=10% VRAM headroom
and a conservative ETC within the deadline. Start training fresh afterward.
The detached pipeline writes persistent logs and seals outputs after
verification. A credential-free remote process guard and hidden workstation
provider-stop guard enforce the lease. No API credential goes to a Pod.

Monitor training every5min and diagnostics every60s, shortening near projected
completion. Report progress, loss, throughput and ETC. Warn on nonfinite or
skipped updates, source/cache/capture mismatch,10min stale events, <10% VRAM
headroom, <5GB disk, or projected deadline/cost overrun.

Transfer about1.02GB of12 models and3 recovery states plus manifests, events,
diagnostics, logs and source identities. Latency also retains raw paired
samples, numerical checks and occupancy diagnostics. Verify archive and
member byte counts/SHA256 locally and run the scientific verifier before
deleting each Pod; then audit remaining account resources. Never delete
unretrieved outputs. The local workspace has sufficient space for archives
and extracted outputs.

## Final latency and verification

Frozen Run041 K050 arithmetic sources and HZ compatibility adapter; RTX5090,
BF16, B1,T2048, full50304 logits, uncached forward with CUDA graphs. Same-model
native SDPA graph reference; unmodified native eager correctness anchor.
One smoke followed by3 fresh scientific processes,64 fixed inputs x7 paired
passes per process (1344 pairs). Every process qualifies all338 blocks:
logit atol.25/rtol.02, relativeL2<=.02, pooled loss delta<=.001. Failures and
raw timings are retained. Training FP16 and runtime BF16 measurements remain
separate. No kernel search or clipping sweep.

Prelaunch results:14 focused tests passed in2.99s, covering matched config/data
order, actual six-layer h-only pressure capture and gradients, kappa=.5 equality
and gate placement, random-parameter replay, checkpoint serialization,
integer ceiling and the three-process latency matrix. The bootstrap suite
passed242 tests in7.82s. All35 run/latency Python files parse. All1,374 frozen
kernel source files passed byte-count/SHA256 checks against Run041. Local
free disk is819.2GiB. The first focused invocation caught an inherited smoke
expectation of kappa=.1; the corrected check derives the requested threshold
from the condition ID, and the rerun passed. No CUDA smoke has run locally;
CPU tests do not qualify CUDA. Exact remote preflight remains mandatory.

Commands: `.venv/Scripts/python.exe -m pytest -q` with this folder's
`test_run044.py` and `latency/test_protocol.py`, using fresh
`--basetemp tmp/run044-tests-002`; the same command with `tests` and
`--basetemp tmp/run044-bootstrap-001`; `19_prepare_latency_sources.py` for the
kernel archive. Reviewed changes from Run041 are the single threshold, worker
count, bounds/collection count, scoped transport names, atomic initialization
upload and report wording. The scientific optimizer/diagnostics are unchanged.

The user confirmed the complete inherited Run041 package without clipping,
then explicitly stated "Launch approved". The approved training Pod is
`5zdd1ayw5cep7i`, one H100 SXM in EUR-IS-3 at USD3.49/hour, created
15:33:09UTC on20September; provider-stop deadline17:33:09UTC. The hidden
workstation guard is armed. The deployment source is
`a4cb5f447d956c7c1d69d14b4036e92f56ad6ef3`, from prelaunch commit
`bdb6244132c51cb2de927bebab840b8af92f0a76`; its receipt is retained. The
first sandboxed guard process could not access the network and exited; it
was replaced with the same scoped guard outside the network sandbox before
deployment. No credential was transferred to the Pod.

Source bundle001 passed its remote SHA256 check and the detached pipeline
started. The first step-zero upload was slow (about1MiB after several minutes);
its transport process was interrupted without stopping the remote pipeline.
Infrastructure-only helper `20_upload_initialization.py` resumed that partial
file with TCP_NODELAY and a4MiB send buffer. The remaining55MB transferred
and verified in10.5s. Both the complete file hash and56,279,344-byte size
match the approved random snapshot; no scientific input changed. Runtime
installation finished and the full validation cache hash matched before
training-cache construction. Training waits for the complete train hash.

Both rebuilt caches passed the full historical hashes. The H100 preflight
passed all six real-data boundaries, with five timed updates of
3.3815/3.3759/3.3718/3.3734/3.3712 seconds (median3.3734s, about622k input
tokens/s). Peak reserved memory61,412,999,168 bytes is72.23% of85,028,372,480.
Full338-block validation took0.407s, checkpoint serialization0.221s, complete
activation diagnostics1.011s and logical diagnostics6.966s. The approved
initial parameter hash, schedule hash, h/z kappa=.5 gates and six h-only
pressure tensors were all verified; no preflight boundary overflowed/skipped.
`prelaunch/remote-preflight-hz-h-ol1-kappa-0p5.json` retains the complete record.

Scientific training started fresh after preflight. At15:53:49UTC it was at
step6/712, loss10.47114, approximately580k input tokens/s including early
checkpoint/validation overhead, with zero overflows and57.2GiB peak reserved.
The refreshed training ETC was43min; the final training diagnostics and
RTX5090 latency remain queued. The full artifact/teardown contract is unchanged.

The live RTX5090 catalog later had LOW stock only in EUR-NO-1, still USD0.99/h.
To avoid a capacity wait and overlap cold compilation, latency Pod
`zf5nwv1pva5d21` was created at16:09:17UTC with the unchanged90min deadline
(17:39:17UTC),40GB container and50GB isolated /workspace. Both scoped provider
guards are armed; the total USD10 cap is unchanged. Source-only bootstrap
`22_build_latency_bootstrap.py` verifies the same1,374 archived files and
bundles the frozen runtime. `latency/00_precompile.py` invokes the existing
K042/K049/K050 extension builders with their unchanged sources and flags;
this is non-evidence compilation only, without a model or timed inference.
`21_prepare_latency_environment.py` verifies the upload hash and runs setup
and compilation detached. Final smoke and all3 complete-qualification timing
processes still run after the actual final checkpoint arrives. This starts
runtime preparation earlier than the initial near-completion estimate while
remaining inside the approved cost/duration envelope.

Runtime preparation attempt001 installed all pins but could not find Ninja
because the virtual environment was absent from PATH. Attempt002 corrected
PATH; it was interrupted after detecting a different extension-cache path
from the final benchmark. Attempt003 used the exact benchmark PATH, CUDA,
extension and Triton cache settings and compiled K042/K049/K050 successfully
in298.6s. Attempt004 precompiles the unchanged K019/K035 dependencies too.
These are infrastructure retries only; the original sources, compiler flags,
scientific inputs and final qualification/timing protocol are unchanged.
All setup and compilation logs are retained for verified retrieval.
Attempt004 also completed successfully (259.0s); all five historical extension
builders now have populated the final benchmark cache. All58 installed package
pins exactly match the frozen Run041 inventory, with a stored verification
record. No scientific timing was collected during preparation.

Training completed all712 optimizer updates without overflow in2430.25s.
All712 OL1 boundary records and four full validation passes verified; final
validation loss is5.5362546204, pooled h/z exact-zero fractions are
0.9984087352/0.9984094443, and R_model is0.0533863908 (analytic ceiling
0.0534714680). Median training throughput was622,090 input tokens/s.
The939,855,159-byte archive and all74 members passed SHA256 verification
locally, followed by the complete local scientific verifier. All12 model
checkpoints and3 optimizer/scaler/RNG states are retained locally
(1,013,222,452 checkpoint bytes). The H100 Pod was then deleted and confirmed
absent; its obsolete workstation guard was stopped. The final checkpoint
and all1,382 source/input identities verified before latency submission.
Local re-verification rewrote only its own verification report with Windows
CRLF line endings. Its normalized bytes and parsed JSON exactly equal the
remote report; the original remote bytes are separately retained under
`transfer/training-verification-remote.json`. All73 other inventory members
remain byte-identical, as recorded in
`prelaunch/training-reconciliation-001.json`.

The smoke and all three full K050 processes qualified. Each scientific
process covered all 338 validation blocks; together they retain 1,344 timing
pairs. Maximum candidate logit absolute error, relative L2 error and pooled
loss difference were all zero against the native eager anchor. BF16 pooled
loss was 5.5384558948 in all modes, kept separate from FP16 training validation.
The 284,950-byte latency archive and all 65 members verified locally before
deletion of the RTX5090 Pod. `17_report.py` generated the final summary and
observation from verified training and latency evidence.

The final account audit confirms both Run044 Pods absent and no Run044
network volume or endpoint. Both workstation guards were stopped. Existing
Run043 Pod `25a3geba9jatml` (USD18.36/h) and the pre-existing 100GB shared
volume `9luykg5yc3` remain untouched. Creation-to-deletion GPU estimates are
USD3.704186 for H100 and USD0.581693 for RTX5090, total USD4.285879. The latest
posted billing contains only USD0.461076 for training and no latency bucket;
it is incomplete and does not supersede the estimate. Storage is excluded
from that GPU estimate. Scoped teardown, live resource audit and the billing
snapshot are retained in `prelaunch/closeout-001.json`.
