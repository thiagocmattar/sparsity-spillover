# K014

K014 is an exploratory Pythia-70M topology-policy search. It leaves the K001
CUDA kernel unchanged and varies only a static layer mask. Every mask is tested
on the predeclared development checkpoints with the complete 338-block
validation gate. A selected topology policy must reuse one mask across kappas.

The six previously inspected untuned checkpoints are forbidden during this
search, so K014 cannot create a new confirmatory holdout claim.
