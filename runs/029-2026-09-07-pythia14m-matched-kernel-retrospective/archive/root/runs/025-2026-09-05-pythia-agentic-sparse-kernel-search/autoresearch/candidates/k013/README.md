# K013: selected Pythia-14M layers-3-through-5 K001 policy

K013 freezes `suffix3`, the winner of K011's predeclared complete-validation
mask search. It applies the unchanged Sakana-derived K001 fused signed-nonzero
compaction kernel to topology-selected linear sites in transformer layers 3,
4, and 5. Layers 0 through 2 remain native. There is no runtime-density or
checkpoint-identity dispatch and no additional pruning.

The search used only six development checkpoints. `suffix1`, `suffix2`, and
`suffix3` passed the unchanged calibration gates over every one of the 338
complete validation blocks for all six checkpoints; `suffix3` won the
predeclared sparse-endpoint geometric-mean rule at 1.058119x. Its six-point
development relation between canonical `R_model` and speedup was positive
(slope 0.161355, Pearson r 0.426759, R2 0.182123). The six interior-kappa
checkpoint conditions remained unevaluated during selection.

Final promotion requires K013's frozen source to re-pass all six development
conditions with 80 paired timing samples per mode and complete validation.
Only then may the fixed evaluator open the six held-out checkpoint conditions.
