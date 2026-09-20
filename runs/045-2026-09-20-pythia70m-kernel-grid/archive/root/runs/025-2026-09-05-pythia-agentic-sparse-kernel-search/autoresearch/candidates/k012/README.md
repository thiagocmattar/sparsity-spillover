# K012: selected Pythia-14M layers-1-through-5 K001 policy

K012 freezes the suffix5 finalist from K011's bounded 12-mask development
search. It applies the unchanged Sakana-derived K001 fused signed-exact
compaction kernel to the topology-selected linear sites in transformer layers
1 through 5 and leaves layer 0 native. There is no runtime density or
checkpoint-identity dispatch and no additional pruning.

At 160 paired samples on the three selection endpoints, suffix5 passed the
fixed numerical gates and measured 1.11165x on A7 kappa=0, 1.11169x on A4
kappa=0.5, and 1.08578x on A7 kappa=0.5. It beat the only screening-tied
finalist, suffix4, on the predeclared high-endpoint geometric mean (1.09865x
versus 1.08123x). Final promotion still requires all six development
checkpoints to pass before the source is frozen for held-out evaluation.
