# Executed methods

The code stores R_model; the paper writes S_model. Both mean the fraction of
logical scalar products in transformer blocks with an exact-zero activation
operand, divided by the block products plus the dense vocabulary projection.
An activation percentage, an MMA bypass percentage and measured latency have
different denominators and cannot substitute for one another.

For L layers, length T, hidden width d, FFN width f and vocabulary V:

```
C_block = T(4d² + 2df) + dT(T+1)
C_model = L C_block + TdV
S_model = sum(exact-zero-operand block products) / C_model
S_max = reachable products under all-zero selected sites / C_model
```

QK and PV count only causal pairs and use a logical OR across their two
activation operands, never double counting. Zero V implies zero context z;
zero scores do not imply zero softmax probabilities. Natural zeros outside the
selected sites can make S_model exceed selected-site S_max. Counts are integers
pooled across all 338 validation sequences and all layers before division.

One-sided threshold retains x when x≥κ; symmetric threshold retains x when
|x|≥κ. Equality survives. Their masks are detached. At h the threshold replaces
GELU; at z it follows attention context concatenation, before W_o. Q/K gates
follow partial RoPE. Pressure reads post-gate tensors, averaging each tensor's
mean absolute value with equal weight across site/layer tensors.

OL1 uses a task-only AdamW step, then a separate pressure correction. For the
bias-corrected Adam task moments and pressure gradient g1:

```
u = m_hat / (sqrt(v_hat) + adam_eps)
w = g1 / (sqrt(v_hat) + adam_eps)
if dot(u,w)<0 and ||u||²>eps:
    w_safe = w - dot(u,w)/(||u||²+eps) * u
else: w_safe = w
r = lambda * ||w_safe|| / (||u||+eps)
s = min(1, b/(r+eps)) if r>0 else 1
theta -= group_learning_rate * lambda * s * w_safe
```

Task gradients are clipped globally to norm 1 before AdamW. Pressure gradients
are unclipped and never enter Adam moments. Projection geometry excludes weight
decay and group learning rates. The cap is not a proof of unchanged loss.
Ordinary L1 instead clips and optimizes the combined task+λL1 gradient.

All paper conditions use 712 updates, seed 1234, length 2048, effective batch
1024, AdamW β=(.9,.95), ε=1e−8, weight decay .1 excluding biases/LayerNorm.
Microbatch/accumulation are 32×32 (14M) and 4×256 (31M/70M/410M). Peak LR is .001
for 14M/31M/70M and .0003 for 410M, with a 10% floor. The exact historical pre-step
cosine/warmup function is retained in training/optimizer.py; update one has LR
zero. Do not replace it with a generic cosine scheduler.

Training uses dynamic FP16 autocast and FP32 parameters/moments, zero dropout,
and Flash SDPA. Logical counters use a separate eager attention pass. BF16 is
used for specialized-kernel inference. The final specialized components are described in kernels/README.md.

Initialization uses small_init σ=sqrt(2/(5d)); residual output matrices use
Wang σ=2/(L sqrt(d)), after the ordinary draw. LayerNorm scales start at one,
biases at zero. Module traversal/RNG backend affects the realized bytes.
Exact input replay requires the omitted canonical random tensors.

The pinned corpus contains 728,374 complete training blocks; the 712×1024
schedule consumes 729,088 and wraps 714 blocks. Complete validation uses all
500 documents: 338 blocks / 692,224 input tokens, with 1,444 tail tokens excluded.
One training seed, one token budget and selected topologies support descriptive
comparisons, not a convergence or scaling-law claim.

The 31M architecture has 30,494,720 parameters, six layers, hidden width 256,
FFN width 1,024, eight heads of width 32, and an untied 50,304-token vocabulary.
Its 11 conditions are Base and T2/Ph and T7/Ph at κ={0,.01,.05,.1,.5}.
They share the recorded random initialization and training order. At every
matched κ, T2/Ph has lower loss than T7/Ph at all three measured latency scales;
T7/Ph reaches a lower minimum latency at each scale. These are observed endpoint
comparisons, not a fitted scaling law or a statement about converged quality.
