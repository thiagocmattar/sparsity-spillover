# Run 055: 31M T7/Ph

Status: implementation and preflight. On 24 September the user confirmed the
five-condition design and explicitly requested immediate, fastest parallel
RunPod execution. This supplies design and launch authorization. The previously
approved complete retention package carries forward. No manuscript edit.

## Question and matched design

Does T7/Ph extend the full-model loss-latency frontier relative to T2/Ph across
14M, 31M and 70M? Train only the missing five 31M T7/Ph endpoints at kappa
0,0.01,0.05,0.1,0.5. Reuse existing matched Base/T2 endpoints and the retained
14M/70M T7/Ph endpoints. New nondominated points support an improved frontier;
domination weakens it. One seed, fixed budget and distinct sessions/size-specific
kernels support a descriptive scale comparison, not a general scaling law.

Pythia-31M has 30,494,720 parameters: six layers, D256, FFN1024, eight heads
of dimension32, vocabulary50304. Architecture revision, canonical random
seed1234 initialization/RNG and MiniPile/tokenizer/cache identities are copied
unchanged from Run054. Initial parameter SHA256 is
731554f7a0e11b2056650f5928fc8dc79db5b0738f49f360ae33a91d631b767e.
No released weights. Model and data seeds1234, same flattened training order.

T7/Ph exactly matches Runs032/034 and the manuscript T7/Ph assignment:
A7-Z-POST; one-sided a,m,h,z; symmetric q_post,k_post,v; equality retained;
Q/K gates follow RoPE; h replaces GELU. OL1 pressure targets only post-gate h,
the equal mean of six layer-tensor means, lambda1 and trust budget1. Task-only
AdamW moments, conflict removal in adaptive geometry, correction after AdamW.

712 updates x1024 sequences x2048 tokens =1,493,172,224 tokens per condition;
five conditions total7,465,861,120. MB4/GAS256, fused AdamW betas(.9,.95), eps1e-8,
weight decay.1 excluding bias/LayerNorm, task clip1, pre-step cosine LR.001 to
.0001 with1% warmup. Dynamic FP16, FP32 parameters/moments, flash SDPA, no dropout
or activation checkpointing. Python3.12/Torch2.11.0+cu128/Transformers5.12.1.
The existing Transformers mapping is retained, not a bitwise NeoX reproduction.

Full validation after step1 and final checkpoint reload covers500 documents,
338 complete2048-token blocks,692224 input tokens,691886 predictions and the
excluded1444-token tail. Complete final activation and eager logical passes
use the same coverage. Pool integer counts before division.

## Retention

Retain model steps0,1,2,4,8,16,32,64,128,256,512,712; optimizer/scaler and
Python/NumPy/Torch CPU/all-CUDA RNG recovery at256,512,712. This gives60 models
and15 recoveries, approximately11GB. Retain all712 gradient/OL1/capture/clip/
overflow boundaries, all named parameter norms, exact/near-zero counts at
0/.001/.01, RMS/L2 and moments at all eight declared sites, six-operation
logical-product counts, and the analytic architecture ceiling. T7 reach is
16109273088/42483056640 products per full sequence (37.919289%). This is analytic
reach, distinct from measured zero products and runtime. No clipping sweep.

## Kernel and timing

Copy the Run054 D256 specialization of Run045 opt073, preserving its fixed
component policy. Pass a/m gates into paired LayerNorm before replacing the
standalone gates. Keep native dense a/m projections, symmetric post-RoPE Q/K/V
gates, dense Flash attention, short-row h/z scalar work with MMA fallback and
the full-vocabulary CUTLASS head. No new optimization search.

RTX5090, BF16, B1/T2048/full logits, CUDA graphs,64 fixed validation inputs x7
paired passes x3 fresh processes per checkpoint. Measure both native and kernel;
plot T7 kernel and the existing Base kernel/PyTorch references. Every process
qualifies all338 blocks: atol.25,rtol.02,relativeL2<=.02,pooled loss delta<=.001.
Replicate1 retains untimed runtime activation moments, row occupancies and
independently checked issued/bypassed MMA and scalar products. Full-model
qualification and explicit fused-gate component tests precede final timing.

## Execution and output

Use five independently preflighted training GPUs concurrently, preferring fast
available H200/H100 capacity, plus a separate RTX5090 timing GPU. Jobs/logs and
checkpoints persist under /workspace and survive SSH disconnection. Read-only
training monitoring every5min; setup/timing every1min; sooner for completion or
nonfinite/skipped updates, stale events10min, missing process, source/cache
drift, disk pressure, or budget/deadline danger. Stop instead of deleting if
artifacts remain unretrieved. Verify archive and every file locally before Pod
deletion, then audit resources. Existing stopped Pod/shared volume are untouched.

Analysis038 will contain the scale figure: Base(kernel/PyTorch), T2/Ph, T7/Ph
at14M/31M/70M,36 execution points from33 checkpoints. Historical coordinates
remain unchanged. PDF only, exact source data, observation and index. No claim
promotion or manuscript edit is authorized. Live allocation, test/ETC evidence,
deadline/cost envelope and transfer receipts belong under prelaunch/.
