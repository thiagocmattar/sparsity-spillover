# Top-k and fixed-threshold motivation

Primary-source review, 8 September 2026. The proposed motivation is supported
with the distinctions below. No manuscript edits or experiments were made.

**Q-Sparse.** Equation 3 selects activation magnitudes at linear-projection
inputs in attention and FFNs. The full procedure also rescales the retained
tensor by its L2 norm and uses a straight-through estimator. Describe the
selection rule without implying that the manuscript reproduces that complete
method. [Q-Sparse, Sections 2.1–2.2](https://arxiv.org/html/2407.10969v3#S2)

**Spark.** Its approximate statistical top-k computes an input-dependent
threshold from the sample mean, standard deviation and a Gaussian quantile.
The operator is one-sided soft thresholding, max(x−theta,0), not
absolute-magnitude hard thresholding. Attention selection concerns token
positions scored using partial query/key dimensions; rejected scores become
negative infinity before softmax. It does not sparsify Q/K/V feature
coordinates as this manuscript does. [Spark, Sections 2.2–3.2](https://arxiv.org/html/2506.06644v2#S3)

**Complexity wording.** “Avoids sorting-based selection costs” is appropriate.
Spark compares its linear-time procedure with naive O(d log d) sorting.
Do not say every top-k algorithm requires sorting or claim a measured
advantage for this paper's thresholds. [Spark, Section 3.1](https://arxiv.org/html/2506.06644v2#S3.SS1)

**This manuscript's design.** Applying the elementwise-nonlinearity principle
to symmetric hard thresholds on internal Q/K/V coordinates accurately names
the implemented intervention. Each training condition fixes kappa; Q/K
thresholding occurs after RoPE. Surviving values and signs are unchanged.
There is no prescribed retained count: activation sparsity may change with
the learned distribution. The masks use the ordinary retained/rejected
derivative, without STE. At positive kappa, hard thresholding is
discontinuous at its boundaries; at zero, the symmetric map is identity.
These statements follow from the manuscript's operational definition.

Frame this as a deliberate design choice that makes sparsity a measured
response to training. Do not claim it is the first extension of nonlinearities
to attention, an approximation to Spark, equivalent to top-k, or an
experimentally established improvement over either prior method. The useful
comparison is fixed threshold versus controlled retention, with different
locations and operators stated explicitly.
