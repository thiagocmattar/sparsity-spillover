# Aligned vocabulary-head training comparison

Both models: h/z8x16 kappa0; head16x16; Ph+head with equal seven-tensor OL1 pressure.
Full500-document/338-block validation;692224 input tokens,1444 excluded tail.

| Head kappa | Loss | Perplexity | Head exact zeros | Empty16x16 tiles | Head RMS | Removed energy |
|---:|---:|---:|---:|---:|---:|---:|
|0.1|5.560725|260.011|39.878%|39.878%|1.387355|11.168%|
|0.5|5.882600|358.741|70.763%|70.763%|1.325602|13.287%|

Paired head-kappa.5 minus.1 loss delta: +0.321875.
These are B32 native activation counts. Actual B1 inference bypass and latency are measured separately in inference/results/REPORT.md.
The two models isolate head threshold at fixed alignment. Historical Run066 comparisons do not isolate alignment alone.
One seed, architecture and budget; no acceptable-quality cutoff or broader speedup is inferred.
