# Run 057: 14M T4 gate sign at LayerNorm outputs

Status: design and launch authorized; local checks passed; provisioning next.

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
