# Run066 - vocabulary-input tiles on the 14M T2/Ph recipe

Status: design, post-hoc inventory and launch approved. The owned two-H100 Pod
has been created; source/input transfer, setup and GPU qualification precede
scientific training. [Launch lease](prelaunch/lease-training-001.json) records
the actual offer, Pod identity, approval and deadline. The original preparation
config's launch flag is historical; explicit user approval authorizes execution.

## Approved question and comparison

Can a tiled threshold at the normalized vocabulary-head input create useful
exact zeros and empty tiles, and does additional activation pressure improve
the loss/structured-sparsity trade-off? This is a quality/sparsity experiment;
head kernels and measured inference speedups are deferred.

Four conditions train from the exact original untrained Run060 parameter bytes:
head gate with Ph, and head gate with Ph plus head-input pressure, each at kappa
0 and 0.1. The h/z and head thresholds are matched within each condition, not
crossed in an eight-condition grid. Existing Run060 endpoints at each threshold
are the controls without the new head gate; their normalized head inputs will
receive the same full-validation diagnostics without further training.

The parent initial parameter SHA256 is
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`;
the matched data-order SHA256 is
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.
These are repository random-pretraining bytes, not released pretrained weights.
Copied runtime source identities before adaptation are in
`prelaunch/derived-sources.json`.

## Exact intervention

`head_in.final` is the learned affine final LayerNorm output, after the last
transformer residual and immediately before `embed_out`:

```text
last residual -> final LayerNorm -> head_input_gate -> dense vocabulary matmul -> logits
```

It is distinct from the existing MLP h site. The block topology registry remains
HZ; condition/realized metadata explicitly records the added head port.
Sequence-local, nonoverlapping 8-token x 16-feature tiles use the same Run060
corner rule: reject the whole tile iff its first-token/first-feature value is
less than or equal to kappa in the actual input comparison dtype. Retained tiles
preserve every signed value. The mask is detached; retained elements have
identity input gradients and rejected elements have zero input gradients.
No additional elementwise gate or straight-through estimator is used. There is
no gate or pressure on vocabulary weights or output logits. Cached decoding
remains outside this sequence-local gate implementation.

Ph remains orthogonal_l1 at post-gate h. The new pressure variant adds the one
post-gate head tensor to the six h tensors. The objective averages tensor mean
absolute values equally: six terms for Ph, seven for Ph+head. Lambda and the
global relative trust budget are both 1; epsilon is 1e-12. Each h tensor's
objective coefficient therefore changes from 1/6 to 1/7. This allocation is an
explicit part of the approved intervention. Pressure uses task-only AdamW
moments, conflict projection and the original post-AdamW correction. Every
boundary checks the exact six- or seven-tensor capture identity.

## Matched training and validation

The pinned Pythia-14M architecture has six layers, D128, FFN512, four attention
heads and V50304. Model/data seeds are 1234. MiniPile/tokenizer revisions, EOS
handling, cache hashes and block order remain those in Run060. Each condition
has 712 updates at global batch1024/T2048: 1,493,172,224 input tokens, with the
same original wrap behavior. MB32/GAS32 is the proposed GPU decomposition.

Training retains dynamic FP16, FP32 parameters/moments, AdamW betas .9/.95,
epsilon1e-8, weight decay .1 excluding bias/norm, task-gradient clip1, zero
dropout, peak LR.001/minimum.0001, original warmup and GPT-NeoX pre-step cosine
schedule semantics. All resolved fields are in `config.yaml`.

Every validation/diagnostic pass covers all500 validation documents and338
complete2048-token blocks:692224 input tokens,691886 next-token predictions,
and the1444-token excluded tail. Loss validation:0,1,32,128,256,512,712.
Full activation diagnostics:0,32,128,712. Model checkpoints:
0,1,2,4,8,16,32,64,128,256,512,712. Complete optimizer/scaler/RNG recovery:
256,512,712. Final checkpoints are retained.

## Diagnostics and interpretation

Measure head pre/post exact/near zeros at0/.001/.01 and the declared threshold,
RMS/L2/moments, sign and removed energy, row nonzero histograms, and empty
1x16/8x16/16x16/32x16 tiles. Verify the corner certificate, per-feature-group
counts and within-tile token-position zero counts. Each full head pass has
88,604,672 activation elements and692,224 aligned8x16 tiles.
Retain matching h/z/block diagnostics, final LayerNorm affine values and named
weight norms. Training events retain task/pressure gradient interaction and
OL1 adaptive direction/trust metrics, which cannot be recovered later.

Head-specific logical products are counted separately from the existing
block-only numerator of R_model. Primary head opportunities use the production
SDPA activation pass; eager/batch1 head counts from legacy block-product
enumeration are stored separately, with native pre/post gate dtypes recorded.
The old HZ analytic ceiling is explicitly
labeled as excluding the new head port. No logical counter is a speedup.

Report full loss/perplexity and deltas to the same-kappa parent. More empty head
tiles with a modest observed loss increase support proceeding to kernel work.
Collapse, little useful tile sparsity or a large quality penalty argue against
it. No numerical acceptable-loss limit was declared; do not claim preserved
quality or label the trade-off acceptable automatically. One seed and this
training budget do not establish broader scaling behavior.

The new head port extends the manuscript's current intervention set and
block-only sparsity numerator in `manuscript/draft/04-methodology.tex`.
Operational head counters stay separately named. No TeX or finding change is
authorized by this run.

## Execution and verification

Focused tests cover placement after affine final normalization, tile algebra,
causality/backward, exact pressure capture/averaging, checkpoint restoration,
complete diagnostic count pooling and retained-control diagnostics. A real14M
T16/MB1 FP32 CPU smoke exercises all four conditions and checks initialization.
These do not establish CUDA memory fit or production ETC. GPU preflights use
the exact full-sequence workload, discard their updated model/optimizer state,
and restart scientific training from original bytes only after all checks pass.

Focused tests39/39 and bootstrap242/242 passed. All four real14M CPU boundaries
passed, with exact initialization, checkpoint logits and causal prefix logits.
Receipts are [checks](prelaunch/checks.json) and [CPU smoke](prelaunch/local-smoke.json).
CUDA qualification attempt001 reached the final preflight summary but failed because
a diagnostic dict shadowed Python's statistics module. No scientific training
started. A reporting-only fix passed the same39 focused and242 bootstrap tests;
attempt002 repeats all four preflights, including exact checkpoint logits/loss,
with unchanged scientific inputs and acceptance criteria. Attempt001 logs,
receipts and calibration checkpoints are preserved under prelaunch/attempts/001.
New qualification receipts/checkpoints use prelaunch/attempts/002.

RunPod deployment, live price/capacity, cost envelope, artifact inventory,
monitoring, warnings, transfer and teardown are in [the launch packet](LAUNCH.md):
two H10080GB GPUs in two waves, four-hour flexible running backstop, USD60 total
ceiling, with a same-scope two-H200 fallback. The approved owned Pod is active at USD6.98/hour. Its original
four-hour deadline remains unchanged across the qualification retry. Any maximum duration is a flexible backstop; normal
completion transfers and verifies artifacts before resource termination.

The deployment bundle is prepared from implementation commit `e745d27f`.
[Source receipt](prelaunch/source-001.receipt.json) records its exact commit,
208,428,798 bytes and SHA256. The bundle and checkpoint weight files remain
outside Git. Source and input uploads were hash-verified; the rebuilt MiniPile
cache matches both approved hashes exactly. Source002 captures the qualification
reporting fix; the initial deployment remains preserved in the source001 receipt.
