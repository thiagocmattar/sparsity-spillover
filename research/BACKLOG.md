# Research backlog

The human selects the research direction. This list records requested work;
it does not authorize implementation, launch or cloud spending.

## Requested next: tile threshold, 14M T2/Ph

Requested on 4 October 2026, after the user closed the P2/Ph comparison. Test
whether training can reorganize representations under a more aggressive tile
gate without degenerating the network. Thresholds are kappa=0 and .1. Measure
activation sparsity, fully zero tiles, and quality; inference latency/speed and
tile-based pressure remain deferred.

Status: concrete design proposed. The user requested further elaboration on
GPU weight-load avoidance and portability across 14M/31M/70M before selecting
tile geometry. [Analysis040](../analyses/040-2026-10-04-tile-gate-hardware-alignment/README.md)
inspects the existing consumers and recommends a per-token K16 gate with
separate inference row grouping. The complete interpretation awaits user
confirmation. No new numbered run or experiment code exists yet.

### Proposed comparison and matched recipe

Two new random-pretraining conditions: T2/tile-gate + Ph at 0 and .1. Compare
with the retained Run058 elementwise T2/Ph conditions at the same thresholds,
using the same complete validation and FP16 evaluation definition. Controls
need not be retrained if initial bytes, cache, data order and recipe match.

- Model: pinned Pythia-14M architecture revision
  7386d9a4ae45aef494a6e704910394def3037fc5; six layers, h width512 and z width128.
  Random small_init/wang_init from config, identical initial parameter hash
  ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57.
- MiniPile train: all 1000000 documents, pinned revision
  18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0; identical tokenizer revision,
  appended EOS, cache identity and realized schedule hash
  f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa.
- Model/data seeds1234; 712 updates, global batch1024, T2048, MB32/GAS32,
  1493172224 input tokens. No continuation from the trained checkpoint.
- Dynamic FP16 with FP32 parameters/moments; AdamW betas .9/.95, eps1e-8,
  weight decay .1 excluding bias/norm, task clipping1, peak LR.001/minimum.0001,
  warmup.01, original GPT-NeoX pre-step schedule mapping; no dropout.
- Gates at h/z only in every layer. h replaces GELU on the W1 output; z is
  concatenated PV before W_o. No gate/pressure at a/m. No double-sided gate.
- Ph remains orthogonal_l1 on post-gate h, unweighted mean of six per-tensor
  mean absolute values, lambda1, relative trust budget1, eps1e-12 and task-only
  AdamW moments. Tile pressure is not part of this first comparison.
- Retain model steps0,1,2,4,8,16,32,64,128,256,512,712 and optimizer recovery
  at256,512,712, including final, as in Run058.

### Exact proposed gate and open geometry

Partition features into fixed contiguous, nonoverlapping tiles within each
token and sequence. Suggested tile is 1 token x16 features; 1x8 or 1x32 are
different designs requiring selection. Never cross batch/sequence boundaries.
Ordinary 8x16 token xfeature tiles are useful occupancy diagnostics but using
their any-element mask as an architectural gate would let earlier tokens
depend on future tokens. The proposed per-token partition preserves causality.

For the raw input X to a gate and tile C, evaluate the user's literal trigger
on the original values once:

```text
drop(C) = any(x == 0 or x < kappa for x in C)
G_tile(X)[C] = 0 if drop(C), otherwise X[C]
```

There is no scan-order dependence, overlapping neighborhood or propagation
from newly written zeros. This is one-sided, not an absolute-magnitude rule.
At positive kappa, equality survives. At kappa=0, the explicit exact-zero
trigger additionally kills a tile containing an original zero, unlike the
elementwise equality-survives contract; this follows the user's stated rule.
Apply the same gate during training and validation/inference.

Proposed backward follows the current detached mask: identity gradient on
surviving tiles and zero input gradient on rejected tiles. No STE, soft gate,
warmup or rescue mechanism is silently added. The post-gate Ph objective also
cannot supply an input gradient through a dead h tile. Representational
capacity alone does not guarantee recovery. Measure the initial and evolving
surviving-tile counts rather than assuming that adaptation will occur.

### Measurements, evidence boundary and later launch

Validate all500 documents, all338 complete2048-token blocks, 692224 inputs and
691886 predicted tokens; report the excluded1444-token tail. Evaluate at
step0,1,32,128,256,512,712 to observe recovery or persistently impaired learning.
Run full activation/tile diagnostics at0,32,128,712; preserve final checkpoint
and original cache identity for later diagnostics. Pool integer counts before
fractions.

Primary table puts exact-zero percentage beside empty-tile percentage at
{a,m,h,z}, per layer and pooled, plus validation loss/perplexity and paired
loss changes against Run058. Count both the chosen gate tiles and the historical
aligned8x16/16x16 diagnostic tiles plus32x16 for the wider retained consumer,
with distinct names and denominators.
Retain pre/post-gate trigger, survival and removed-energy counts at h/z,
exact/near-zero counts at eps0,.001,.01, RMS/L2, named weight norms and
per-boundary task/pressure/OL1 gradient geometry. Logical products are auxiliary
opportunity counters, not speed; no clipping sweep or latency benchmark.

Support: increased empty-tile structure with ongoing learning, surviving branch
activity and a measured quality trade-off. Refute successful adaptation in this
budget: persistent branch collapse, stalled loss, nonfinite behavior or a large
quality penalty. Report the actual loss/perplexity delta; no numerical definition
of an acceptable quality penalty has been selected. One seed, two thresholds
and a fixed horizon cannot rule out adaptation with another training recipe.

This proposes a new grouped operator relative to the elementwise gates in
manuscript/draft/04-methodology.tex (sparsification interventions); it tests the
quality/structured-zero trade-off. Operational Ph stays unchanged. No TeX edit
or speedup claim is requested. After design confirmation, implement focused
gate/gradient/causality/counting tests and the bootstrap suite, calibrate the
production workload, and present a current RunPod execution/cost envelope.
The earlier GPU price and spending cap do not become a new launch quote.

Post-hoc inventory is proposed above; before launch confirm whether any other
measurements are needed. In particular, gradient interaction must be retained
during training and cannot be reconstructed from final weights alone.

## Deferred tile pressure

Requested and explicitly deferred on 3 October 2026. Investigate pressure and
gating organized by activation tiles, with the aim of creating blocks that
hardware can skip during training or inference. The 4 October request resumes
tile gating only, as described above; a tile-based pressure objective remains
deferred.

The requested next gate uses an any-zero/any-below-kappa trigger, while a
group-magnitude trigger remains a different deferred method. Tile pressure
definition and hardware kernel alignment remain undecided.
Retain row/tile occupancy in the preceding gate experiment where practical.

Evidence and limitations: [gate and tile diagnosis](../analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/README.md).

## Approved design: 14M gate sign comparison

The latest request narrows the study to T4 with two pressure recipes, RunPod,
and two or at most three thresholds per intervention. The proposed P-o = Pall
interpretation (a,m,h,z pressure) was included in the design approved by the
user's “Toca o barco” on 3 October 2026.

The [concrete proposed design](../analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/PROPOSED-14M-T4-DESIGN.md)
uses two thresholds, holds h/z gates fixed, and varies the sign rule at a/m.
Implementation belongs to [Run 057](../runs/057-2026-10-03-pythia14m-t4-gate-sign/README.md).
The user's subsequent explicit RunPod/AFK authorization covers launch,
monitoring, retrieval and analysis through figures/results within the recorded
responsible envelope.
