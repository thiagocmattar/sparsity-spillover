# K011: bounded Pythia-14M K001 layer-mask search

K011 inherits the unchanged Sakana-derived K001 fused signed-exact compaction
kernel and varies only a static six-layer mask. Native linears execute outside
the mask. There is no runtime density or checkpoint-identity dispatch and no
additional pruning.

The fixed 12-policy space contains prefixes and suffixes of lengths one through
five plus even and odd layers. Development selection uses 14M A7-OL1 kappa=0
as the correctness sentinel and the A4/A7 kappa=0.5 development endpoints as
the speed objective. Any finalist must pass all 16 development inputs for all
six development checkpoints before being frozen for untouched interior-kappa
evaluation.
