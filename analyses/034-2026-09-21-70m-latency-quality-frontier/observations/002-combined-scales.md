# Validation loss and full-model latency across 14M and 70M

## Question and method

Where do the retained 14M and 70M measurements lie on the same validation-loss
and full-model-latency axes?

This figure overlays the 40 checkpoints from Analysis033's focused 14M figure
and the 26 checkpoints from this analysis's 70M figure. It preserves their
exact coordinates, recipe grouping and latency conventions. There is no
normalization, axis break or connection between model scales. The plot omits
naive L1, GeLU-to-ReLU and post-hoc clipping, as in its two source figures.

Source script: [`../02_combined.py`](../02_combined.py).
Exact points, source hashes, backend identities and drawing checks:
[`../data/combined-figure.json`](../data/combined-figure.json).
Sources are Analysis033 `data/three-panel.json` and Analysis034
`data/figure-data.json`; Analysis027 supplies recipe colors and line styles.

Coverage: ordinary final-checkpoint FP16 loss over 338 complete 2,048-token
blocks from all 500 MiniPile validation documents, excluding the 1,444-token
tail. Each latency uses RTX5090, BF16, batch one and full output logits,
with 64 inputs x seven passes x three processes. The 14M values retain their
four historical sessions; the 70M values retain Run045 and Run047. Measurements
are not pooled or rescaled across sessions or model sizes.

## Figure and caption

[Publication PDF](../figures/02-14m-70m-latency-quality.pdf).

**Validation loss and full-model latency on Pythia-14M and Pythia-70M.**
Circles denote 14M (40 checkpoints) and squares denote 70M (26 checkpoints).
T2/Ph is blue, T7/Pall orange and other recipes gray. Hollow markers identify
Base; vertical dotted lines mark its validation loss at each scale. Lines
connect settings within the same recipe and model size. Validation loss is
linear and latency logarithmic, with lower values preferred. Tick labels retain
milliseconds. Latencies use K050 at 14M and opt073 for the 70M
intervention recipes. The 70M Base marker uses native PyTorch; the 14M Base
marker uses K050, as explicitly identified in the legend.

## Results and limits

The displayed 14M points occupy losses 5.110-6.038 and latencies 0.459-0.652 ms;
70M occupies losses 4.100-5.390 and latencies 1.088-2.741 ms. The common axes
show the lower validation losses and higher absolute latencies at 70M for
the measured workload. They also expose the quality costs within each scale.

The Base latencies use different implementations: 0.651573 ms for K050 at 14M
and 1.611048 ms for native PyTorch at 70M. The optimized 70M Base value is not
substituted into this view. Different kernels, timing sessions and one training
seed limit interpretation; this is not a controlled estimate of model-size
effects or the causal benefit of sparse execution. The logarithmic latency axis
makes variation within the 14M cluster easier to see while retaining the full
range of both scales. No manuscript text or other PDF changes.
