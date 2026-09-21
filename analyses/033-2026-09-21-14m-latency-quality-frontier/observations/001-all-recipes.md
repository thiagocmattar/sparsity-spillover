# Latency and validation loss across 14M manuscript recipes

## Question

How do full-model latency and validation loss vary across the trained 14M
recipes shown in the main manuscript?

## Method and coverage

Source: Analysis028 `data/full-trained-results.json`. At the user's request,
the four naive-L1 appendix checkpoints and the GeLU-to-ReLU control are
excluded, leaving 40 checkpoints across nine recipes. OL1 remains included.
Analysis027 supplies the manuscript colors and typography; Analysis030's
`02_plot.py` and `23-70m-quality-sparsity-native-latency.pdf` supply the
highlighted lines, markers and white marker edges.
Generating script: [`../01_plot.py`](../01_plot.py).
Exact coordinates, exclusions and hashes: [`../data/frontier.json`](../data/frontier.json).

Coverage is Base, four OL1 T1/Ph pressure settings, and five settings
each for T2/Ph, T4/P0, T7/P0, T4/Ph, T7/Ph, T4/Pall and T7/Pall. Dose settings
are lambda=.05,.1,.5,1 for T1 and kappa=0,.01,.05,.1,.5 for the threshold
recipes. Training uses one seed, 712 steps and the shared data schedule.
Ordinary final-checkpoint validation covers all 338 complete 2,048-token
blocks from 500 MiniPile documents, excluding the 1,444-token tail.

Latencies retain the historical K050 geometric means, including specialized
Base execution at 0.651573 ms. These are not native-PyTorch timings. The
workload is RTX5090, BF16, batch one, 2,048 tokens and full vocabulary logits;
each checkpoint has 1,344 synchronized host samples (64 inputs x seven passes
x three independent processes). Sessions contribute Run029:30, Run033:5,
Run041:4 and Run044:1 checkpoints. No normalization or new timing measurement
is performed. Run048's separate execution controls are not mixed into this grid.

## Figure and caption

[Publication PDF](../figures/01-14m-latency-quality-frontier.pdf).

**Latency and validation loss across Pythia-14M manuscript recipes.**
Full-model K050 latency versus final validation loss for 40 trained checkpoints
across nine recipes. Lower values are preferable on both axes. T2/Ph is blue
and T7/Pall orange, using the manuscript's stronger lines and markers; other
recipes appear in gray. Base is emphasized with a hollow marker and a vertical
dotted loss guide. Only Base and the two highlighted recipes have legend labels;
point callouts are omitted. Naive L1, the ReLU control and the Pareto overlay
are excluded as requested. Lines connect separately trained settings and do
not represent measured interpolation.

## Result

T2/Ph at kappa=.1 has loss 5.151098 and latency 0.573427 ms. T4/Ph at kappa=.05
has higher loss (5.195590) and lower latency (0.561801 ms), while both are below
Base loss (5.208583). At kappa=.1, T4/Ph takes 0.529309 ms with loss 5.228687.
The fastest displayed checkpoint, T4/Pall at kappa=.5, takes 0.459476 ms but
has loss 6.037987. These illustrate the measured quality-latency trade-offs.

## Interpretation limits

This comparison is descriptive. Timing sessions and hosts differ, so small
gaps may change ordering under remeasurement; no training-seed uncertainty
estimate is available. It does not isolate sparsity's causal contribution or
establish performance for cached decoding or other batch/sequence shapes.
The source still retains the naive-L1 and ReLU results; excluding them is a
presentation choice, not a claim that the displayed set exhausts all measured
14M recipes. Post-hoc clipping is outside this trained-recipe figure.

The manuscript and its existing figures remain unchanged. No finding is
promoted by this analysis.
