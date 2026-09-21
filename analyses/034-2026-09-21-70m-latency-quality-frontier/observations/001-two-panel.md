# Sparsity and quality versus latency at 70M

## Question and method

How do sparsity and validation loss relate to the measured 70M latencies,
highlighting T2/Ph and T7/Pall in the same style as the focused 14M figure?

Both panels show the same 26 checkpoints across six recipes, drawn from
[Analysis030](../../030-2026-09-20-70m-t2-optimized-endpoint/observations/001-later-70m-endpoint.md).
The only excluded trained control is GeLU-to-ReLU. There are no post-hoc
clipping curves, ceiling guides or conditional execution ablations.
The source script is [`../01_plot.py`](../01_plot.py); exact coordinates and
hash-verified provenance are in [`../data/figure-data.json`](../data/figure-data.json).

Each non-Base recipe contains kappa=0, 0.01, 0.05, 0.1 and 0.5. Twenty-five
displayed points retain Run045's timings; the T2/Ph kappa=0.5 endpoint retains
Run047's qualified timing. No sessions are pooled or rescaled. Base uses the
original native PyTorch latency, 1.611048 ms; other points use opt073. This
backend convention matches the manuscript's 70M figure. It differs from the
14M counterpart, where the Base marker uses the specialized implementation.

Coverage: ordinary final-checkpoint FP16 validation over all 338 complete blocks
from 500 MiniPile documents, excluding the 1,444-token tail. Sparsity is 100
times pooled integer zero-product counts divided by all full-model products,
including the dense vocabulary head. It is not the fraction of zero activations.
Host latencies are geometric means from RTX5090, BF16, batch one, 2,048 tokens
and full logits: 64 inputs x seven passes x three independent processes.

## Figure and caption

[Publication PDF](../figures/01-70m-sparsity-latency-quality.pdf).

**Sparsity, latency and quality on Pythia-70M.**
(a) Model-wide logical sparsity and full-model latency.
(b) Validation loss and full-model latency for the same
26 checkpoints. T2/Ph is blue, T7/Pall orange and other recipes gray. Hollow
markers and dotted vertical and horizontal guides denote Base; its latency
uses native PyTorch, while the other points use the optimized implementation.
Lines connect separately trained threshold settings within each recipe.
The later T2/Ph endpoint retains its own timing session without rescaling.

## Results and limits

At kappa=0.5, T7/Pall has 40.602% sparsity and latency 1.275788 ms; T2/Ph has
15.427% sparsity and latency 1.214368 ms. Their validation losses are 5.215976
and 4.838034. This comparison holds kappa fixed, not quality or intervention
placement, and uses different timing sessions.

The lower-loss T2/Ph points, at kappa up to 0.1, have losses 4.129-4.193 but
latencies 1.956-2.504 ms, exceeding native Base's 1.611 ms. Thus the targeted
recipe's favorable validation loss does not imply a speedup over native Base.

The optimized stack includes dense improvements as well as sparse skipping.
These curves do not isolate the causal contribution of zeros, establish
statistical equivalence between timings or generalize beyond one training
seed and the measured full-sequence workload. No manuscript text is changed.
