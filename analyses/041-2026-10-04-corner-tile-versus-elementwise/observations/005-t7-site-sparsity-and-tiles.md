# T7/Ph: what changes in seven-site sparsity and empty tiles?

Question: compare the approved14M T7/Ph corner operator at kappa0/.1 with the
matched retained scalar T7/Ph checkpoints, distinguishing element zeros from
fully empty8x16 tiles.

Method and coverage: Run062 keeps Run032's random initial parameter bytes,
ordered-block schedule, seed1234,712 updates, FP16 autocast/FP32 parameters,
optimizer and h-only orthogonal L1 pressure. Its signed a/m/h/z and magnitude
q/k/v gates reject an8x16 tile when its first corner is <=kappa (absolute corner
for q/k/v). Admitted interiors retain their values; equality also changes.
Q/K gates follow RoPE, h replaces GELU and z precedes W_o. The complete operator
changes, so this does not isolate geometry. Scalar occupancy is a separate
native FP16 replay of c36/c39, without optimization or clipping. Canonical scalar
quality retains the original full-validation value; replay discrepancies are
under.00002. Logical-product counts retain each original diagnostic source.

All500 validation documents:338 complete2048-token blocks,692224 inputs and
691886 predictions, excluding1444 tail tokens. Pool exact integer counts across
all six layers; QKV tiles remain within each head. Both remote/local training
verifiers pass, including every corner certificate; the archive and148 members
have verified SHA256/byte identities. No final layer/site is completely empty.

Caption: two rows are kappa0/.1; seven columns are a,m,h,q_post,k_post,v,z.
Blue bars are scalar gates; hatched purple bars are corner tiles. Each panel
places element-zero percentage beside fully empty8x16-tile percentage with a
shared0–100% axis. Empty-tile and element-zero counts are distinct estimands.

Results: empty a/m tiles rise from effectively0% to55.017%/54.471% at kappa0
and53.415%/53.388% at kappa.1. The h/z comparison is:

| kappa | h zeros: scalar → tile (%) | h empty tiles: scalar → tile (%) | z zeros: scalar → tile (%) | z empty tiles: scalar → tile (%) |
|---:|---:|---:|---:|---:|
| 0 | 93.475 → 61.764 | 1.513 → 61.764 | 64.394 → 48.389 | .003 → 48.389 |
| .1 | 99.133 → 92.842 | 57.705 → 92.842 | 96.820 → 92.461 | 52.100 → 92.461 |

At kappa.1, empty q/k/v tiles become11.378%/10.404%/22.259%, versus effectively
zero in the scalar controls. At kappa0 they remain effectively zero. Greater
tile occupancy structure can therefore coexist with fewer element zeros.
Canonical loss rises.501244/.597707, and perplexity65.077%/81.795%, so the
structure comes with a substantial fixed-budget quality penalty.

Caveats: one seed and two thresholds; complete-operator comparison. Finite
learning and surviving activity do not establish equal quality, redistribution,
portability, DRAM avoidance or runtime gain. R_model is logical opportunity.

Artifacts: [PDF](../figures/05-t7-site-sparsity-and-tiles.pdf),
[wide table](../results/t7-comparison.md), [CSV](../results/t7-comparison.csv),
[counts, layer ranges and source hashes](../results/t7-training.json).
Source script: [03_t7_training.py](../03_t7_training.py). PDF rendered and
visually inspected; earlier figures are preserved.
