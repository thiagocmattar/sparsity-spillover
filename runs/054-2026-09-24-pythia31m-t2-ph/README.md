# Run 054: 31M Base and T2/Ph

Status: complete. All six conditions finished 712 updates, full validation and
diagnostics; all 18 RTX 5090 timing processes passed numerical qualification.
The complete cohort and transferred artifacts were verified locally. All six
task-created Pods, including the replaced host, are terminated. The user
authorized fastest available RunPod execution and retained the full package.
See `prelaunch/parallel-fleet-001.json` and `prelaunch/compute-closeout.json`.

## Question and approved comparison

Add a middle scale to Analysis 034 Figure 02: does the approximately 30M model
extend the measured full-model latency versus validation-loss frontier?
There are **six training conditions**, not 52: Base and T2/Ph at
`kappa = 0, 0.01, 0.05, 0.1, 0.5`. The earlier 52 count interpreted the initial
ellipsis as increments of 0.01; the user's explicit five-value set supersedes it.

The standard pinned architecture is `EleutherAI/pythia-31m-deduped`, revision
`b7782556ba7adfb4730d9bda7d12aa44d88fa132`: 30,494,720 parameters, six layers,
width 256, FFN 1024, eight heads of width 32, vocabulary 50,304.
All conditions replay one CPU-generated seed-1234 small_init/wang_init snapshot.
No released weights are loaded. The snapshot parameter SHA-256 is
`731554f7a0e11b2056650f5928fc8dc79db5b0738f49f360ae33a91d631b767e`.

The recipe matches Runs 041/044 (14M) and 043/046 (70M): random initialization;
the same pinned MiniPile/tokenizer cache and flat seed-1234 training order;
712 updates of 1,024 sequences of 2,048 tokens; 1,493,172,224 input tokens per
condition; fused AdamW, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1 excluding
bias/LayerNorm, task gradient norm clipping at 1; pre-step cosine LR 0.001 to
0.0001 with 1% warmup; dynamic FP16 with FP32 parameters/moments; SDPA flash;
zero dropout and no activation checkpointing. MB4/GAS256 matches 70M; the
14M MB32/GAS32 decomposition has the same flattened example order.
These are the established Transformers recipe mappings, not bitwise GPT-NeoX
framework reproduction. Python 3.12, Torch 2.11.0+cu128, Transformers 5.12.1.

Base retains GELU and has no gates or activation pressure. T2/Ph uses topology HZ:
one-sided h and z gates, retaining values equal to kappa; h replaces GELU;
orthogonal L1 pressure acts only on post-gate h with weight 1 and step budget 1.
Gate sites and pressure sites remain independently declared. No post-hoc clipping.

Full validation means all 500 documents: 338 complete blocks, 692,224 input
tokens, 691,886 next-token predictions, and the excluded 1,444-token tail.
Validation runs after step one and after reloading the final checkpoint.
Final canonical activation and eager logical-product passes cover all 338 blocks.
Integer numerators/denominators are pooled before computing fractions.

## Retention and implementation

`config.yaml`, architecture JSON and initialization metadata pin the inputs.
`02_train.py` reuses the audited Run 004 lifecycle and Run 043 h-only boundary,
with run-local 31M initialization and complete recovery serialization.
`03_verify.py` checks scientific identities, full validation, every optimizer
boundary, diagnostic coverage and every retained artifact hash. `04_monitor.py`
reads persistent progress. `05_preflight.py` measures complete production-shaped
boundaries including staging, optimizer and OL1 work.

Retain model steps 0,1,2,4,8,16,32,64,128,256,512,712 and optimizer, loss scaler,
Python/NumPy/Torch CPU/all-CUDA RNG states at 256,512,712. This is 72 model
snapshots and 18 recovery states, approximately 13.2 GB before small metadata.
Retain every gradient conflict/OL1 boundary metric, clipping behavior and
overflow history; final per-layer weight norms; exact/near-zero counts at
0/0.001/0.01, RMS/L2 and pooled statistics for all eight declared activation
sites; logical-product counts and architecture ceiling. HZ analytic reach is
4,026,531,840 / 42,483,056,640 products per full input sequence (9.47797%).
This ceiling is not measured sparsity or runtime speedup.

`07_package.py` creates an explicit source/initialization archive and optional
cache archive from committed source. `12_check_inputs.py` verifies uploads.
`08_pipeline.py` runs six remote preflights, checks the measured forecast against
the deadline, then trains and verifies six conditions and invokes the latency
grid. `10_seal.py` seals terminal files; `11_retrieve.py` checks the downloaded
archive, every file, training cohort and latency qualification before teardown.
The external workstation stop guard is `13_local_stop_guard.ps1`.
`15_parallel_train.py` implements the approved parallel launch: one condition per
GPU, concurrent preflights, and a verified condition before retrieval. All six
training conditions use H200; only execution placement changes. MB4/GAS256,
initialization, order, pressure, precision and validation remain unchanged.

## Kernel and latency contract

`latency/` specializes **Run 045 opt073**, the 70M kernel behind the referenced
Analysis 034 figure. It does not substitute the newer Run 051 implementation.
The fixed policy is paired LayerNorm, native dense a/m projections, short-row
h/z SIMT with native-order MMA fallback (M8/N256/K16, limit eight entries),
dense Flash attention (M64/N128, four warps, head dimension 32, token-major
output), and the CUTLASS 128x128x32 full-vocabulary head. The z K256 mask has a
partial 32-bit word; scalar counts use output width 256. No optimization search.
Copied sources and their parent/port hashes are in `latency/port-provenance.json`;
each measurement records its full runtime source inventory.

Final measurements require RTX 5090, BF16, B1/T2048, full 50,304-token logits,
CUDA graphs, 64 fixed validation inputs, seven paired timing passes, and three
fresh processes per checkpoint. All six record native and kernel graph timing;
the requested plot will show five T2/Ph kernel points plus Base kernel and Base
PyTorch (CUDA-graph replay), preserving the original figure's execution contract.
Every process qualifies against native eager on all 338 validation blocks:
pointwise atol 0.25 / rtol 0.02, relative L2 <=0.02, pooled loss difference <=0.001.
An unqualified point is retained and flagged, never silently presented as valid.
Replicate one also records untimed BF16 activation moments, h/z row occupancy,
issued/bypassed MMA and scalar products checked against independent oracles.
These issued-work counters remain distinct from canonical logical opportunities.

After verified retrieval, a new numbered analysis will preserve the existing
14M/70M points and add this model, with a PDF and observation Markdown following
Analysis 034 Figure 02. Same GPU class/workload, but different sessions, one seed
and only three sizes: this supports a descriptive frontier, not a fitted scaling
law or evidence that every intervention improves latency. Neither a speedup nor
a loss improvement is assumed. The manuscript will not be edited in this task.

## Local verification and launch

See `prelaunch/launch-proposal.md` for checks, measured local fit, billable
envelope, artifact transfer and monitoring. Calibration checkpoints and timings
are infrastructure evidence only and must never enter the final figure.

## Parallel execution record (2026-09-24)

The user's fastest-completion instruction superseded the sequential proposal and
authorized launch. Six H200s ($4.59 per GPU-hour) train one condition each, with
a separate Community RTX 5090 ($0.69/hour) for timing. The initial fleet rate is
$28.23/hour; the fleet cap is $200 and the external stop deadline is 19:01 UTC.
No scientific batch, optimizer, validation, initialization or data-order input
changed. `prelaunch/parallel-fleet-001.json` records the initial deployment.

The EUR-IS-4 H200 required about 13.3 seconds per T2 update; the US workers
required about 5.3 seconds. Cache assembly measured under 0.01 seconds, so moving
the cache would not solve the observed bottleneck. A replacement H200 in US-NC-1
passed its preflight and reached about 400,000 tokens/s. The original kappa-0.1
attempt was stopped with SIGTERM at 13:47:59 UTC (inherited SIGINT was ignored).
Its original manifest, events and checkpoints are retained unchanged; the
terminal pipeline and `prelaunch/retired-training-attempts.json` identify this
infrastructure failure. The replacement restarts the same canonical inputs.
This placement decision used throughput, not model quality. The extra host is
removed after all of its evidence has been copied and hash-verified.
The retired Pod was terminated at 13:53 UTC after 1,342,224,756 bytes of training
and preflight artifacts plus its control logs were verified locally. Its control
archive omitted `pip-freeze.txt`; the failed attempt still records its pinned
runtime and source identities. Retrieval now explicitly includes that file for
every completed training/timing Pod.

Initial cache upload won a race against a pinned remote rebuild, and its archive
and four files were verified before use. Training Pods use a compact source
package; unused archived CUDA headers delayed extraction on the first host.
The latency compile retry added the virtual environment's `bin` directory to
PATH so the installed Ninja executable was discoverable. These were setup
retries before scientific measurements, and their logs are retained.

For this parallel execution, `prelaunch/retrieve-sealed.py` retrieves terminal
Pod archives into isolated directories, checks archive and per-file hashes,
then merges only verified evidence. It supersedes the sequential retrieval
entry point. `prelaunch/verify-cohort.py` calls the unchanged scientific verifier
for all six completed conditions and requires retained evidence for the recorded
infrastructure interruption. `prelaunch/stage-final-latency.py` transfers only
verified final-model files to the RTX 5090; optimizer history is retained locally.
`latency/03_final_grid.py` runs three fresh timing processes per condition under
an exclusive GPU lock. This overlaps timing and retrieval with remaining
training without concurrent benchmarks on the same device.

The cross-run output is Analysis 037. Its collector requires six verified final
training endpoints, 18 fully qualified timing processes, one GPU UUID, one
runtime source identity and exact final-checkpoint hashes. The plot preserves
all 68 historical execution coordinates and adds seven for the new model.

## Verified results and closeout

| Condition | Validation loss | Kernel ms | PyTorch ms | R_model (%) |
|---|---:|---:|---:|---:|
| Base | 4.565514 | 1.106255 | 1.023861 | 0.000018 |
| T2/Ph, kappa=0 | 4.688476 | 1.020591 | 1.068473 | 8.588719 |
| T2/Ph, kappa=0.01 | 4.701284 | 1.017293 | 1.071115 | 8.732630 |
| T2/Ph, kappa=0.05 | 4.702921 | 0.969383 | 1.072476 | 9.056343 |
| T2/Ph, kappa=0.1 | 4.757208 | 0.908418 | 1.071218 | 9.259301 |
| T2/Ph, kappa=0.5 | 5.220418 | 0.681118 | 1.067923 | 9.465328 |

Latencies are geometric means across three fresh processes, each containing
448 paired samples per backend. PyTorch means CUDA-graph replay, matching the
historical figure. R_model is a count-pooled logical-product fraction, not a
runtime gain. The training-loss curve is not substituted for final validation.

The [updated PDF](../../analyses/037-2026-09-24-14m-31m-70m-latency-quality/figures/01-14m-31m-70m-latency-quality.pdf)
and [observation](../../analyses/037-2026-09-24-14m-31m-70m-latency-quality/observations/001-combined-scales.md)
contain the cross-scale interpretation and limits. The exact endpoints are in
`artifacts/verification.json` and Analysis037's `data/31m-results.json`.

All 72 scientific model snapshots and 18 recovery states are retained locally:
13,175,935,092 checkpoint bytes. The verified evidence archives total
16,741,091,156 bytes including preflights, the interrupted infrastructure attempt
and latency evidence, before the small separate control-log archives. Checkpoint
weights, recovery binaries and duplicate transfer archives are excluded from Git.
All six final models share the canonical initialization and data schedule; no
optimizer update was skipped. Every final condition has 48 canonical activation
rows, weight statistics, logical counts, and all 712 optimizer-boundary records.
Every timing condition has 42 BF16 runtime activation rows and 12 h/z work and
occupancy records over the complete 338-block validation set.

All task Pods were absent in the provider audit at 14:50:52 UTC. The pre-existing
stopped Run052 Pod and 100 GB shared volume are unchanged. All task stop guards
are stopped. The conservative compute estimate is USD45.63, using nominal rates
from Pod creation through deletion (including provisioning time), excluding
storage. Posted task billing was USD25.00 but covered only the buckets through
14:00 UTC; it is not the final invoice. See the complete closeout JSON.

One kappa-zero SFTP upload stalled before measurement began. Only that transfer
was restarted using an SSH stream; archive and model-file hashes were checked
again. No completed scientific measurement was discarded or repeated.

Verification: 11 focused checks and the 242-test bootstrap suite passed before
launch; production-shaped preflights ran concurrently on assigned GPUs; all six
completed attempts passed the unchanged scientific verifier locally; all 18
timing processes passed full-block qualification and exact checkpoint checks.
The PDF was rendered and visually inspected, with all 75 execution points in
range and the historical PDF unchanged. No manuscript or finding was promoted.
