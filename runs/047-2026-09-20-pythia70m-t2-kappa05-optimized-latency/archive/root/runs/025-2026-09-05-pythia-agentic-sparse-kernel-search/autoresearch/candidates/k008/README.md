# K008: bounded 70M K001 layer-mask search

K008 inherits the unchanged K001 fused signed-exact compaction kernel and
varies only a static six-layer mask.  Native linears execute outside the mask;
there is no runtime density dispatch or additional pruning.  Prefixes and
suffixes of lengths one through five plus even/odd masks form the complete
12-policy development search.

The first search uses the first seven frozen development inputs of 70M
A7-OL1 `kappa=0.5`, which include all inputs that failed the prior 16-input
K001 gate.  Any finalist must subsequently pass all 16 development inputs and
be frozen in a new candidate before held-out or complete-validation use.
