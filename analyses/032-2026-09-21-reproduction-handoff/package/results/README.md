# Evidence and figure observations

These are compact retained measurements, not new experiments. Source paths and
hashes identify records in the author archive. They are provenance labels and
are not files that the portable scripts need to load. JSON was compacted without
changing numerical values; MANIFEST.json hashes the release bytes.

| File | Question, coverage and interpretation |
|---|---|
| endpoints.json | Quality/sparsity and session-labeled latency for all 84 final trained conditions, one seed each; 338 validation blocks per condition. Sparsity is a percentage in these display rows; zero/model product counts are integers. |
| 14m-figure.json | The 41-point main 14M display and clipping controls; four L1 appendix settings are outside this main plot. |
| operation-latency.json | Where do skips save latency? Six conditional effects at one 14M T7/Pall κ=.5 checkpoint. Whiskers span process-pair differences, not confidence intervals; effects are not additive. |
| pressure-placement.json | Where does h-only pressure change exact zeros? Thirty 14M T4/T7 conditions, seven sites and six layers, with raw per-cell zero/total counts. |
| ol1-geometry.json | Does pressure saturate its budget? Retained 712-update traces and summaries; plotted 14M settings are distinguished from supporting 70M records. |
| base-training.json | Base-model loss and gradient trajectories at three sizes; centered nine-update smoothing is retained. Fixed budget does not establish convergence. |
| 70m-controls.json | Same-checkpoint opt073/native-h,z substitutions in their own session. Conditional difference includes fusion, layout and inspection. |
| 70m-sessions.json | The later T2 κ=.5 measurement and repeated Base/κ=.1 references; no cross-session pooling. |
| kernel-structure.json | Scalar sparsity versus projection/attention MMA bypass and Base-relative speed. Padding and scalar substitutions are counted; bypass is not FLOPs saved. |
| clipping.json | All 300 retained 14M post-hoc points (30×10), including unfavorable tails. Clipping is an evaluation intervention on fixed checkpoints. |

`python scripts/reproduce.py figures` writes nine PDF reconstructions to
`reproduced/`, using these inputs. Source: `scripts/reproduce.py`. The original
paper figures remain in `paper-figures/`, with identical source bytes.

The reconstructed quality plots show loss/latency against count-pooled logical
sparsity, with recipe lines and dotted Base/ReLU clipping paths. The three-panel
plot joins shared 14M/70M quality recipes to conditional operation savings. Layer
maps use a common 0–100% color scale; rounded/colored cells do not prove exact
all-zero tensors. OL1 plots show medians across steps/thresholds, not confidence
intervals. Base trajectories show smoothed loss and pre-clipping gradient norms.
The kernel scatter is descriptive, and the clipping plot retains full loss tails.

All plots are derived views of one-seed measurements. They do not introduce
new training evidence, quality-controlled speed comparisons or causal site
attributions beyond the explicitly matched ablations. The full mathematical
and runtime boundaries are in docs/METHODS.md and docs/REPRODUCE.md.
