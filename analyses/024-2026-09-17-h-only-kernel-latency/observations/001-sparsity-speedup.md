# 001 - Model-wide sparsity and final K050 speedup

**Question.** Where do the new A7+OL1(h) models lie relative to retained A4/A7
pressure families under the final frozen K050 kernel?

**Method and coverage.** `../01_reduce.py` reads 35 historical Run029 K050
checkpoints and five new Run033 checkpoints. Only the latter were benchmarked
now. Each point pools 1,344 raw paired host-latency ratios geometrically: 64
fixed validation inputs, seven passes, three fresh processes. BF16, batch one,
T=2048, uncached causal inference, all 50,304 logits; recurring packing and
zero detection are timed, equal input staging is excluded, and setup is
separately recorded. Each process passes numerical checks on 338 complete
blocks from all 500 validation documents (1,444-token tail excluded).
S_model is reconstructed from the retained pooled FP16 integer logical-product
counts. All 40 plotted checkpoints qualify. Sources and hashes are in
`../data/results.json`; the complete table is `../TABLE.md`.

**Caption and legend.** Model-wide logical sparsity versus full-model K050
speedup relative to the same checkpoint's native PyTorch SDPA CUDA graph on
RTX5090. Blue/orange distinguish A4/A7. Solid open-marker curves have no OL1;
dashed filled curves use all active pressure sites; dotted square/diamond
curves use h-only OL1. Lines connect kappa=0, 0.01, 0.05, 0.1, 0.5 in order.
The black star is A0 and gray crosses are the nine A1-H controls. Whiskers are
the range of three process geometric means, not confidence intervals. The
horizontal dashed line is break-even. No point annotations or fitted trend
line are shown. A7 h-only is measured in Run033; every other point is reused
from Run029, on another physical RTX5090/host.

**Result.** New A7 h-only speedups increase from 1.206029x to 1.783869x over
the five thresholds. At kappa=0.5, h-only/all-site A7 have observed speedups
1.783869x/1.783175x with S_model 16.663640%/27.482684%. The measured runtime
endpoints are close despite the substantially different logical-product
opportunity. The reused A4 h-only endpoint is 1.599928x at 10.227390% S_model.

**Limits.** This is descriptive checkpoint evidence, not a universal sparsity
speed law, an equivalence test, or a decomposition of direct Q/K/V pressure.
Only one retained trained model exists per condition. Run029 GPU UUID is
`d77f736c-1ebe-6277-380f-b544e7271074`; Run033 is
`3ce3f8e5-ad63-9457-8104-f384b1f446c4`. Native-relative pairing retains the
reference definition, but host/GPU differences can affect ratios as well as
absolute latency. Recorded CPU thread counts are 96 and 64 respectively.
No new A0 or skip-control ablation was measured. Training quality remains in
Analysis023 with its explicit validation-pass conventions. No manuscript
claim has been promoted from this observation.

**Source and output.** `../02_plot.py` generates
`../figures/.archive/01-14m-k050-sparsity-speedup.pdf` from the reduction. The final PDF
was rendered and visually checked with embedded TrueType fonts.
