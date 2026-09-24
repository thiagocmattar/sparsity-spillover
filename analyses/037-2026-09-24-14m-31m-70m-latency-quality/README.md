# Pythia 14M, 31M and 70M: loss versus full-model latency

User-requested extension of Analysis 034 Figure 02 with the six Run 054
checkpoints (30,494,720 parameters): Base and T2/Ph at kappa 0,0.01,0.05,0.1,0.5.
Base appears under both kernel and PyTorch execution. Historical coordinates
are preserved exactly, yielding 75 execution points for 72 checkpoints.

Training and final latency measurements are in progress. No preflight result
will be substituted for a trained endpoint. The manuscript is unchanged.

Quality is canonical final-checkpoint FP16 validation over all 338 complete
2048-token blocks from 500 MiniPile validation documents; 1444 tail tokens are
excluded. Timings use RTX 5090, BF16, B1/T2048 and full logits, 64 inputs x seven
passes x three fresh processes. The new kernel specializes Run 045 opt073 for
width 256, FFN 1024 and head dimension 32. All processes must pass full-block
qualification against native eager. Training runs concurrently on six H200s.

`01_collect.py` verifies the new cohort, checkpoint identity, raw timing reduction,
process coverage and numerical qualification. `02_plot.py` reproduces the
original figure's recipe colors, logarithmic axes and old coordinates, adding
triangular markers for 31M. Per-process latency dispersion remains in the data.
No timing rescaling, cross-size curve fit or extrapolation is introduced.

Different execution sessions and scale-specific kernels, a single training seed,
and sparse sampling of model size limit any scaling-law interpretation. The
scientific result and final observation will be written after verified retrieval.
