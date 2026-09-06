# K006: frozen architecture-shape portfolio

K006 is a zero-overhead setup-time dispatch over the independently qualified
K001 and K005 CUDA kernels.  It installs the selected inherited module directly;
there is no extra model-forward wrapper or runtime density test.

The selection was frozen from development primitive timings on the RTX PRO
4500: use K005 for `(K,N)=(128,128)` (Pythia-14M z) and `(2048,512)`
(Pythia-70M h), and K001 everywhere else.  The map depends only on projection
shape, not checkpoint, family, kappa, observed R_model, or validation input.
Both inherited kernels preserve identical signed exact compaction and FP32 FMA
order, so K006 introduces no extra sparsity.  A three-way paired full-model
probe must determine whether this portfolio actually improves K001 at the same
checkpoint/R_model.
