# Run 057: 14M T4 gate sign at LayerNorm outputs

Status: all eight conditions completed and verified locally; the owned Pod is
deleted. Tables, four PDF figures and the one-seed conclusion are retained.

The user approved the [concrete design](../../analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/PROPOSED-14M-T4-DESIGN.md)
with “Toca o barco” on 3 October 2026, after its eight-condition matrix and
diagnostic/retention inventory were presented. The proposed interpretation of
spoken P-o as Pall executes: pressure at a,m,h,z. Ph targets h only. This is
design approval. The subsequent instruction to continue while AFK through
analysis, figures and conclusions, with full authorization specifically to
launch RunPod experiments using account funding, authorizes execution and
closeout without another confirmation. The launch packet is [LAUNCH.md](LAUNCH.md).

The hypothesis is that replacing unilateral a/m gates with bilateral gates
preserves useful negative activations and improves validation quality. Compare
fresh unilateral and bilateral a/m conditions at kappa=0 and 0.1 under Ph and
Pall. h/z stay unilateral, all six layers, A4-Z topology. Bilateral kappa=0 is
identity at a/m. Threshold equality survives. Gates replace GELU at h; a/m are
after learned LayerNorm affine parameters, z is concatenated PV before W_o.

Random step-zero bytes, seed 1234, pinned Pythia-14M architecture/tokenizer,
MiniPile cache/order, 712 updates, batch 1024, sequence 2048, dynamic FP16,
FP32 parameters/AdamW moments, LR and OL1 recipe match the approved design.
Released or trained weights are not initialization. The local replay input is
ignored by Git; its provenance and required hash are tracked.

Full validation uses all 500 documents: 338 complete blocks, 692,224 input
tokens, 691,886 prediction tokens, excluded tail 1,444. Pressure averages means
uniformly across the actual post-gate tensors (6 for Ph, 24 for Pall), audited
at every boundary. Gradient interaction is retained during training.

Collect final pre/post a/m sign counts and moments, all-site gate rejection
energy, exact/near zeros, a/m/h/z row and aligned tile occupancy, learned LN
affine values, all named weight norms, full logical counters/analytic ceiling,
and full validation. Retain all declared model checkpoints and three recovery
states, including final. Tile pressure is [deferred](../../research/BACKLOG.md).

Support: bilateral a/m lowers final full-validation loss within each matched
pressure/threshold pair, accompanied by preservation of negative activation
energy. Refutation: no improvement or degradation. One seed and joint a/m
changes cannot isolate a versus m or prove a LayerNorm mechanism. Equal kappa
does not match sparsity. Logical opportunity and empty tiles are not speedup.
No manuscript edits are authorized. The affected scientific question is the
site-dependent quality/sparsity tradeoff in the paper's experimental study.

The later explicit launch authorization supersedes the pending confirmation
noted in the initial design record. It does not authorize manuscript changes.

## Execution record

The deployed source is commit `27489414d7c32766d02475d9064c8e86aedab505`.
Secure Pod `6auyq9zlfezk72` started at 2026-10-03T16:11:59.867Z with eight H100
80GB GPUs, USD27.92/hour, in AP-IN-2. The initial EUR-NO-2 placement request
failed without creating a resource; the globally eligible request succeeded at
the same rate. Provider-stop deadline: 2026-10-03T19:11:59.867Z. The independent
local guard and credential-free remote guard preserve unverified evidence.

Infrastructure recovery inside `prelaunch/attempt-001/` restored the required
sparse checkout after a checkout timeout and resumed the step-zero upload with
a larger SSH transfer window. No scientific training had started and no
scientific input was changed. Rather than uploading the six-GB token cache over
the slow connection, the pinned MiniPile/tokenizer builder reproduced both
canonical cache hashes on the Pod in 686.723 seconds. The random step-zero
snapshot was also checked against its declared byte count and SHA256.

All eight exact-workload preflights passed before the training barrier opened:
five timed updates after one warmup, median 3.386--3.609 seconds per update,
72.24--72.84% reserved memory, no overflow or skipped update, full 338-block
validation, checkpoint round trip, activation and logical-product diagnostics.
Actual CPU allocation is 224 vCPU and 2,013GB RAM. The setup and placement
record is retained in the lease, source receipt, preflight JSONs and control logs.

## Verified result and closeout

The bilateral-minus-unilateral full-validation loss differences are:

| Pressure | kappa | Loss difference | R_model difference (percentage points) |
|---|---:|---:|---:|
| Ph | 0 | -0.085038 | -3.7852 |
| Ph | .1 | -0.054435 | -3.6034 |
| Pall | 0 | +0.018569 | -3.8487 |
| Pall | .1 | +0.126668 | -1.3589 |

Negative loss differences favor bilateral. Ph improves quality while losing
logical opportunity. Under Pall, unilateral has both lower loss and higher
logical opportunity at each tested cutoff. Thus the hypothesis receives
pressure-dependent support, without establishing a universal bilateral benefit
or a causal LayerNorm mechanism. One seed and jointly changed a/m remain the
interpretation limits. No inference speedup was measured.

The [complete tables](results/tables.md), [source-hashed numerical export](results/results.json)
and [observation with figure captions and caveats](observations/001-gate-sign.md)
include activation energy, LN moments/affine arrays, tile occupancy and retained
gradient interaction diagnostics. All four PDF pages were rendered and visually
checked. Tile pressure remains deferred; no manuscript narrative was changed.

Every condition completed 712 optimizer updates without overflow/skips. Remote
verification passed, then all 529 returned files (8,122,095,599 payload bytes;
7,525,455,638-byte compressed archive) passed archive/member hashes locally.
Local `03_verify.py` passed the complete cohort. The 96 model checkpoints and
24 optimizer/scaler/RNG recovery states are retained locally, including final.

Pod deletion returned HTTP204 and provider inventory confirmed absence at
2026-10-03T17:39:36Z. The local deadline guard was disarmed afterward.
The owned Pod ran approximately 87.60 minutes: estimated GPU charge USD40.764,
plus about USD0.018 storage, below the USD90 envelope. Posted billing was still
partial (USD11.188), so that subtotal is not the final bill. See the
[compute closeout](results/compute-closeout.json). A concurrently running CDR
experiment Pod, the unrelated EXITED Run052 Pod and the shared network volume
were left untouched and are outside this run's cost and lifecycle.
