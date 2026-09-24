# Evidence and figure observations

These are compact retained paper measurements. Scientific condition names replace
private experiment IDs; checkpoint, initialization and schedule hashes preserve
identity. Timing sessions retain distinct descriptive labels. Only fields used
by the paper companion are included; measured values are unchanged.
PROVENANCE.json records source hashes and MANIFEST.json hashes release bytes.

| File | Question, coverage and interpretation |
|---|---|
| endpoints.json | Quality/sparsity and session-labeled latency for all 95 final trained conditions, one seed each; 338 validation blocks per condition. Sparsity is a percentage in these display rows; zero/model product counts are integers. |
| scale-frontier.json / scale-evidence.json | Current Figure 5 coordinates, 31M endpoint/delta table, initialization/order identities and analytic ceilings. Base is represented with both kernel and PyTorch backends. |
| 31m-diagnostics.json / 31m-timing-processes.json.gz | Ordinary final validation and integer logical counts for all 11 conditions; complete numerical qualification and raw timings for all 33 processes. |
| 14m-paper-figure.json | Current Figure 1: identical 40 checkpoint keys in the two scatter panels; same six controlled operation effects. |
| 14m-t2-execution-controls.json | Same-checkpoint h,z 2-by-2 execution intervention; zeros and logits unchanged, 12.9% joint reduction, 57.9% of the native gap. |
| 14m-figure.json | The broader 41-point 14M context and clipping controls; four L1 appendix settings are outside this main plot. |
| operation-latency.json | Where do skips save latency? Six conditional effects at one 14M T7/Pall κ=.5 checkpoint. Whiskers span process-pair differences, not confidence intervals; effects are not additive. |
| pressure-placement.json | Where does h-only pressure change exact zeros? Thirty 14M T4/T7 conditions, seven sites and six layers, with raw per-cell zero/total counts. |
| ol1-geometry.json | Does pressure saturate its budget? Retained 712-update traces and summaries; plotted 14M settings are distinguished from supporting 70M records. |
| base-training.json | Base-model loss and gradient trajectories at four sizes; centered nine-update smoothing is retained. Fixed budget does not establish convergence. |
| 70m-controls.json | Same-checkpoint final/native-h,z/skips-off substitutions in their own session. Conditional difference includes fusion, layout and inspection. |
| 70m-sessions.json | The later T2 κ=.5 measurement and repeated Base/κ=.1 references; no cross-session pooling. |
| kernel-structure.json | Scalar sparsity versus projection/attention MMA bypass and Base-relative speed. Padding and scalar substitutions are counted; bypass is not FLOPs saved. |
| clipping.json | All 300 retained 14M post-hoc points (30×10), including unfavorable tails. Clipping is an evaluation intervention on fixed checkpoints. |

`python scripts/reproduce.py figures` writes ten PDF reconstructions to
`reproduced/`, using these inputs. Source: `scripts/reproduce.py`. Exact original figure assets are in `figures/`;
captions are in [the manuscript](../main.pdf).

The reconstructed quality plots show loss/latency against count-pooled logical
sparsity, with recipe lines and dotted Base/ReLU clipping paths. The three-panel
plot reconstructs the 40-point 14M quality/latency/logical-opportunity panels
with the six conditional operation savings. Layer
maps use a common 0–100% color scale; rounded/colored cells do not prove exact
all-zero tensors. OL1 plots show medians across steps/thresholds, not confidence
intervals. Base trajectories show smoothed loss and pre-clipping gradient norms.
The kernel scatter is descriptive, and the clipping plot retains full loss tails.

All plots are derived views of one-seed measurements. They do not introduce
new training evidence, quality-controlled speed comparisons or causal site
attributions beyond the explicitly matched ablations. The full mathematical
and runtime boundaries are in docs/METHODS.md and docs/REPRODUCE.md.

Ready-to-read tables `endpoints.csv`, `scale-frontier.csv` and `31m-results.csv`
are included and checked byte-for-byte against the offline reconstruction.
The first uses kernel latency; the scale table labels both Base backends;
the 31M delta table uses PyTorch Base, matching the manuscript.
