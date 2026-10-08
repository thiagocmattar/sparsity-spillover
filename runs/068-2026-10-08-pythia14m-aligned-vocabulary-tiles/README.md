# Run068 — aligned vocabulary-head tiles

Status: design confirmed on 8 October 2026; implemented and locally qualified;
awaiting separate launch approval. No scientific training or Run068 cloud spend
has occurred. Development checkpoints and timings are not experimental results.
See [the launch packet](LAUNCH.md) and [local checks](prelaunch/checks.json).

## Approved question and conditions

Does training a head gate aligned to the M16/K16 vocabulary consumer create
useful skippable sparsity, and how does raising the head threshold affect loss?
Exactly two fresh Pythia-14M models are approved:

| Condition | Head kappa | Head tile | h/z kappa | h/z tiles | Pressure |
|---|---:|---|---:|---|---|
|t2-hz0-head16-ph_head-kappa-0p1|0.1|16 tokens x16 features|0|8 tokens x16 features|Ph+head|
|t2-hz0-head16-ph_head-kappa-0p5|0.5|16 tokens x16 features|0|8 tokens x16 features|Ph+head|

The two new models isolate the head threshold at fixed alignment and pressure.
Comparisons with Run066 are historical context: its head-kappa0/hz-kappa0 model
also changes the head threshold, while its head-kappa.1 model has hz-kappa.1.
Neither comparison isolates tile alignment alone. One seed, architecture,
training budget, GPU and full-sequence workload limit generalization.

## Exact operator, site and kernel contract

The head gate follows the learned affine final LayerNorm, immediately before
`embed_out`. Each sequence has independent nonoverlapping [token,hidden-feature]
tiles. Reject the whole tile iff its native FP32 top-left value is <=kappa.
Retain every signed interior value when the corner is >kappa. Equality rejects.
The detached mask gives live tiles identity input gradients and dead tiles zero
input gradients. No straight-through estimator is used.

```text
Final residual -> learned affine final LayerNorm -> FP32 head16x16 corner gate
               -> FP16 vocabulary projection H_head @ W_out -> all V logits
```

The gate acts on input activations, not vocabulary weights or output logits.
V=50304 and all logits are computed and written. Training uses the native dense
projection and backward. The custom consumer is inference only.

The adapted CUDA kernel reads one native FP32 post-gate corner per16x16 tile
before casting inputs to FP16. It uses M16/N256/K16 and ascending-K FP32 MMA
accumulation with FP16 outputs. N128/N512 are component checks, not a tuning
search. A negative or zero interior value at token8 is not another certificate.
FP16 rounding cannot decide support. Full16x16 gate emptiness equals consumer
K16 bypass, with independently measured instruction counts. Empty tiles do not
remove the cost of writing the complete vocabulary output.

Both conditions retain h/z gates, but all transformer projections execute native
dense kernels in the endpoint comparison. Only the head physically skips work.
Cached decoding and integration with earlier BF16 block kernels are out of scope.

## Matched training

Six layers, D128, FFN512, four attention heads and V50304. Replay the same
untrained random parameter bytes used by Run060/066; no released or trained
weights are loaded. Model/data seeds1234. Initial parameter SHA256:
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`.
Training schedule SHA256:
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.

The pinned MiniPile/tokenizer revisions, EOS handling, verified int32 caches and
wrap schedule match Run066. Each model has712 updates at global batch1024/T2048:
1,493,172,224 input tokens. Production decomposition is MB32/GAS32 per GPU.
Use dynamic FP16 and FP32 parameters/moments; task-only fused AdamW moments,
betas.9/.95, epsilon1e-8, weight decay.1 excluding bias/norm, task clip1, zero
dropout, peak LR.001/minimum.0001, eight warmup updates and GPT-NeoX pre-step
cosine semantics. The entire optimizer boundary is skipped on overflow and
reported; skipped updates disqualify the matched endpoint.

Both use `orthogonal_l1`, lambda1, relative trust budget1, epsilon1e-12, task
moment preconditioning, conflict projection and the post-AdamW correction.
Six post-gate h tensors plus one post-gate head tensor are equally averaged;
each tensor coefficient is1/7. Gate and pressure configuration remain separate.

Every validation/activation pass covers all500 documents and338 complete2048
blocks:692224 input tokens,691886 predictions,1444 excluded tail tokens.
Loss steps:0,1,32,128,256,512,712. Full activation diagnostics:0,32,128,712.
Model checkpoints:0,1,2,4,8,16,32,64,128,256,512,712. Optimizer/scaler/RNG
recovery:256,512,712. All final checkpoints are retained.

## Diagnostics and endpoint comparison

The user confirmed retaining pre/post exact/near-zero counts at0/.001/.01 and
the declared thresholds; RMS/L2/moments, signs/removed energy, row occupancy,
1x16/8x16/16x16/32x16 occupancy, corner certificates, feature-group and within-tile
position counts, final LayerNorm affine values, named weight norms, training
gradient-conflict/OL1 directions/trust metrics, and logical-product counts.
Pool integers before dividing. Head opportunities stay separate from block-only
R_model and its HZ ceiling. No clipping sweep or token-prediction file is planned.

After terminal training verification and local hash-verified retrieval,
`inference/01_prepare.py --bundle` seals the two actual endpoints. A separate
RTX5090 compares native dense, custom no-skip and custom sparse paths in six
fresh processes, three per model;64 fixed validation inputs x7 randomized paired
passes. Full qualification covers all338 blocks in every process. Preserve
atol.25/rtol.02 per logit, relative L2<=.02 and pooled loss delta<=.001; report
exact equality as well. Both complete head paths and projection-only graphs
qualify on every timed operand.

Measure projection-only, gate/cast/projection and full-forward boundaries with
synchronized host replay and separately replayed CUDA events. These boundaries
overlap and must not be added as a latency decomposition. Static weight casting
and packing, staging, softmax, token selection and loss are outside timing;
recurring activation preparation is included in head/full paths. Retain raw
timing pairs, input indices, all numerical checks, work counts, setup/memory,
environment/source/cache/checkpoint identities and failed attempts.

`18_report.py` reduces only verified terminal training evidence; the inference
reducer requires verified local transfer. Final tables/observations/PDFs will
belong here. Greater useful bypass with a favorable measured loss-latency
tradeoff supports the question; loss degradation without enough sparsity or
latency savings argues against it. Increasing kappa is monotonic for fixed
activations, not necessarily after adaptive training. The custom no-skip
baseline was slower than cuBLAS in Run067; alignment alone guarantees no speedup.

This informs the manuscript's quality-compute-runtime frontier. The operational
head port and strict corner rejection execute; no manuscript edit or finding
promotion is approved.

## Implementation and development evidence

Training derives from frozen Run066 and its audited Run004 primitives. The
new condition resolver independently declares block/head thresholds and tile
sizes; checkpoint restoration and head coverage verification use those fields.
The inference consumer derives from Run067 with native M16 support extraction.
Original executed folders remain unchanged.

242 bootstrap tests and44 focused tests pass in separate Python processes.
Both real14M CPU smokes restore exact initialization/checkpoint logits and
preserve causal-prefix logits. Final WSL RTX5070Ti development tag003 passes
four reduced FP16 OL1 boundaries,63 CUDA component checks,2028 full-logit
comparisons across both models/all338 blocks and768 standalone graph comparisons.
Every full/head comparison is exact. These models had one CPU mechanics update;
their losses and laptop timings do not answer the scientific experiment.

Retained tag001 passed components and the first complete full-forward check,
then the projection-only wrapper rejected an old M8 support-array shape. The
wrapper and tile metadata were corrected; tag002 repeats the complete cohort.
Tag003 repeats the cohort after separating the trained endpoint inventory and
adding the retained reduced-CUDA training receipt; numerical criteria stay fixed.
A combined pytest invocation also exposed historical `run_config` module-name
collision during collection; bootstrap/focused suites pass in isolated processes.
Neither issue required changing numerical criteria or scientific inputs.

Production MB32/GAS32 resource fit and ETC will be checked on the approved
training GPU before fresh initialization and scientific execution. Nothing is
launched while approval is pending.
