# Observations

| ID | Observation | Evidence |
| --- | --- | --- |
| O001 | L1N-at-h versus untargeted attention near-zero mass for GeLU and ReLU. | valid - ten matched conditions, full validation, verified checkpoints |
| O002 | Individual q_post, k_post, v, and m trajectories versus h for GeLU and ReLU. | valid - terminal counts plus full-validation post-hoc m from ten hash-verified checkpoints |
| O003 | Post-W_o/pre-residual attention-output near-zero mass versus h. | valid - full-validation post-hoc diagnostic from ten hash-verified checkpoints with exact zero-dropout boundary equivalence |
| O004 | Epsilon-1e-2 version of the h-versus-attention-average trajectory. | valid - count-first reduction of stored full-validation integer counts |
| O005 | Epsilon-1e-2 version of the q_post/k_post/v/m sitewise grid. | valid - count-first reduction of stored terminal and post-hoc full-validation integer counts |
| O006 | Epsilon-1e-2 version of the post-W_o/pre-residual trajectory. | valid - count-first reduction of the verified full-validation post-hoc diagnostic |
