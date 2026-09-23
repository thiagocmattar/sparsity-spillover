# Validation loss and full-model latency across 14M and 70M

Historical 68-point version. The 23 September additive update is documented in
[observation 006](006-recent-kernel-overlay.md); all original coordinates remain.

## Question and method

Where do the retained 14M and 70M measurements lie on the same validation-loss
and full-model-latency axes?

This figure overlays the 40 checkpoints from Analysis033's focused 14M figure
and the 26 checkpoints from this analysis's 70M figure. Each Base checkpoint
is shown under both kernel and native PyTorch execution, giving 68 execution
points for 66 unique checkpoints. Existing measurements and recipe grouping
are preserved. There is no
normalization, axis break or connection between model scales. The plot omits
naive L1, GeLU-to-ReLU and post-hoc clipping, as in its two source figures.

Source script: [`../02_combined.py`](../02_combined.py).
Exact points, source hashes, backend identities and drawing checks:
[`../data/combined-figure.json`](../data/combined-figure.json).
Sources are Analysis033 `data/three-panel.json` and Analysis034
`data/figure-data.json`; Analysis027 supplies recipe colors and line styles.
Analysis024 `data/results.json` supplies the paired 14M Base timings. Its K050
value matches the existing 14M marker exactly, and both timings reproduce the
geometric means of three qualified replicates. The 70M reference pair comes
from the existing qualified Run045 Base record. Canonical validation loss is
held fixed within each Base pair.

Coverage: ordinary final-checkpoint FP16 loss over 338 complete 2,048-token
blocks from all 500 MiniPile validation documents, excluding the 1,444-token
tail. Each latency uses RTX5090, BF16, batch one and full output logits,
with 64 inputs x seven passes x three processes. The 14M values retain their
four historical sessions; the 70M values retain Run045 and Run047. Measurements
are not pooled or rescaled across sessions or model sizes.

## Figure and caption

[Publication PDF](../figures/02-14m-70m-latency-quality.pdf).

**Validation loss and full-model latency on Pythia-14M and Pythia-70M.**
Circles denote 14M and squares denote 70M. The plot shows 64 intervention
points and four Base execution references. T2/Ph is blue, T7/Pall orange and
other recipes gray. Hollow Base markers denote kernel execution and smaller
filled Base markers denote native PyTorch; both are shown for each model.
The 14M Base markers nearly coincide. Vertical dotted lines mark Base validation
loss at each scale. Lines connect settings within the same recipe and model
size. Both axes are logarithmic, with lower values preferred; tick labels show
validation loss and milliseconds. Kernel execution uses K050 at 14M and
opt073 at 70M.

## Results and limits

The displayed 14M points occupy losses 5.110-6.038 and latencies 0.459-0.655 ms;
70M occupies losses 4.100-5.390 and latencies 1.088-2.741 ms. The common axes
show the lower validation losses and higher absolute latencies at 70M for
the measured workload. They also expose the quality costs within each scale.

Both implementations are explicit for each Base checkpoint:

| Model | Kernel latency (ms) | PyTorch latency (ms) | Timing session |
| --- | ---: | ---: | --- |
| 14M | 0.651573 (K050) | 0.655442 | Run029 |
| 70M | 2.732112 (opt073) | 1.611048 | Run045 |

The two 14M references differ by less than 0.6%, so the smaller filled marker
appears inside the hollow marker without displacing either point. At 70M,
the optimized kernel's Base execution remains slower than native PyTorch.
Different kernels, timing sessions and one training seed limit interpretation;
this is not a controlled estimate of model-size
effects or the causal benefit of sparse execution. The logarithmic latency axis
makes variation within the 14M cluster easier to see while retaining the full
range of both scales. No manuscript text or other PDF changes.
