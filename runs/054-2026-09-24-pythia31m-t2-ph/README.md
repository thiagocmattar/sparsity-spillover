# Run 054: 31M Base and T2/Ph

Status: design and launch approved. The user replaced the sequential budget
proposal with fastest available RunPod execution and concurrent conditions.
Six H200 GPUs and a separate Community RTX 5090 are provisioned; remote setup
and preflights precede scientific training. See `prelaunch/parallel-fleet-001.json`.
The user reconfirmed the full diagnostic/checkpoint retention package.

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
