# 001: Complete 70M T2/Ph figure

**Question.** Where do the five completed 70M T2/Ph checkpoints fall in the
existing quality-sparsity and full-model latency figure?

**Method and coverage.** Preserve the original 22 endpoints and 20 fixed-control
clipping settings from Analysis024. Add the four Run043 checkpoints and Run046's
kappa=0.5 endpoint, with gates at h/z, h-only orthogonal L1 pressure and lambda=1.
Quality is ordinary final-checkpoint validation loss, not the BF16 kernel
qualification loss. Model-wide sparsity is 100 times the ratio of pooled integer
zero-product counts to full-model logical products. All 338 complete blocks from
500 documents are evaluated; the 1,444-token tail is excluded. No T2 post-hoc
clipping was performed.

Each new endpoint's frozen K050 latency is the geometric mean of 1,344 paired
host timings (three fresh processes, 64 inputs, seven passes), BF16 on RTX5090,
batch one, 2,048 tokens, all 50,304 logits and uncached CUDA-graph inference.
All 15 new processes pass full-validation numerical qualification. The original
22 timings and native Base line remain unchanged. Run043 and Run046 use later
GPU sessions; a visual comparison to the historical Base line is not a matched
native speedup experiment. Same-checkpoint native ratios are retained in JSON.

**Result.** The figure now contains 27 unique trained checkpoints and seven
recipe styles in both panels. The complete T2/Ph grid approaches its 15.443084%
logical reach ceiling; the kappa=0.5 endpoint has validation loss 4.838034,
model-wide sparsity 15.426970% and specialized latency 1.690154 ms. Its separately
paired native ratio is 1.001063x, with process ratios on both sides of unity.
It therefore shows no clear runtime advantage over its own native implementation.
The added series follows the same blue dashed style as the 14M T2/Ph figure.

| kappa | Validation loss | Model-wide sparsity (%) | K050 latency (ms) |
|---:|---:|---:|---:|
| 0 | 4.128669 | 14.629632 | 2.772560 |
| 0.01 | 4.149913 | 14.699409 | 2.728571 |
| 0.05 | 4.192501 | 15.029591 | 2.451620 |
| 0.1 | 4.182317 | 15.212165 | 2.302983 |
| 0.5 | 4.838034 | 15.426970 | 1.690154 |

**Legend and caption.** Quality, sparsity and execution on Pythia-70M.
(a) Validation loss and (b) full-model latency versus model-wide logical sparsity
for 27 trained checkpoints: Base, GeLU-to-ReLU, T2/Ph, and T4/T7 with Ph or Pall.
Every pressure recipe has kappa=0,.01,.05,.1,.5. Dotted curves in (a) are retained
post-hoc clipping of the fixed Base/ReLU controls; high-loss tails exceed the
focused view. Vertical guides mark the T2/T4/T7 analytic ceilings. In (b), the
Base marker and horizontal line use native PyTorch/SDPA (1.664 ms); other markers
use the frozen original 70M port. The original 22 checkpoints share Run035's
RTX5090 session; T2 uses later Run043/Run046 sessions. The workload is batch one,
2,048 tokens and full logits. Later optimization results are separate.

**Caveats.** One seed and one training budget; this joint gate/pressure
intervention does not isolate either component. Logical sparsity is not removed
FLOPs or speedup. Cross-session timing differences are descriptive. The graph
connects the ordered thresholds, not a continuous fitted response. Historical
measurements, clipping tails and the native Base reference are unchanged.

**Sources.** `../01_collect.py`, `../02_plot.py`, `../03_verify.py`, and
`../data/70m-quality-sparsity-native-latency.json` with 59 direct source hashes.
The canonical PDF is `../figures/23-70m-quality-sparsity-native-latency.pdf`;
the manuscript figure and supplementary data are exact copies. No broader
finding or manuscript narrative is promoted by this update.

**Display revision.** At the user's request, the rotated "Post-hoc" annotation
was removed. Both clipping curves, all data, axes and other labels are unchanged.
