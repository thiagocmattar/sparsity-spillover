# OL1 implementation, geometry and saturation audit

Sources: Analysis 020 `01_audit_logs.py`, `geometry.py`, `test_geometry.py` and
`data/log-audit.json`. All 336 relevant source-file references in the 54 manifests
match current source bytes (or the manifest-declared LF normalization). This
includes run-local FP16 boundary code, imported boundary helpers, logging code
and shared pressure/optimization modules. Original run files are unchanged.

Run 009 optimizer_boundary.py accumulates task/pressure gradients separately,
unscales both, rejects nonfinite boundaries, clips only the task gradient, steps
AdamW, then calls src/sparsity_research/pressure.py:apply_ol1_correction. Run 018
uses Run 017's boundary through its frozen import; Run 019 has the same sequence
with exact capture-name verification. The task first/second moments are updated
and bias-corrected before u and w are formed. Pressure is unclipped and has no
first-moment buffer. Projection uses c<0 and q>eps; coefficient c/(q+eps).

On parameters with both gradients and initialized task moments, r=lambda*norm(w_tilde)/
(norm(u)+eps), s=min(1,b/(r+eps)) for r>0, otherwise 1. Each group receives
-lr*lambda*s*w_tilde after AdamW. Decoupled decay is outside this geometry; group
learning rates follow the pooled cap. All 34 OL1 runs have b=1, eps=1e-12; local
weights are .05/.1/.5/1, multisite weight is 1. Eligible tensors: 65 for local
14M h-only, 69 for 14M/70M multisite and 285 for 410M multisite.

The appendix gives the ideal half-space/norm-ball proof. CPU tests check aligned,
orthogonal, conflicting, opposing and zero vectors, cap boundary, positive global
rescaling, the true-task-gradient counterexample and the small-norm safeguard.
No loss-preservation theorem or empirical robustness sweep is claimed.

All 24,208 OL1 boundaries are retained, no steps skipped. 7,283 caps bind (30.0843%).
The test uses the actual condition r>0 and b/(r+eps)<1 and verifies logged s and
final ratio. Norms are pre-learning-rate on eligible tensors. norm(w_tilde) was
not stored directly; it is recoverable as r*(norm(u)+eps)/lambda, a derived value.
Every first boundary has zero LR; positive-LR counts are retained separately.

For a fixed expanded set, sum_7/(7L) vs sum_7/(4L) is positive global scaling;
it disappears ideally when both corrections saturate. Adding targets changes
the direction, which does not disappear. Actual trajectories at untested weights
are not implied by this fixed-state property.

| Condition | Phase | Cap / boundaries | Median correction/task ratio |
|---|---|---:|---:|
| 14M:A1-H-OL1:0.05 | all | 0/712 | 0.029871 |
| 14M:A1-H-OL1:0.05 | early_1_142 | 0/142 | 0.026144 |
| 14M:A1-H-OL1:0.05 | late_570_712 | 0/143 | 0.045419 |
| 14M:A1-H-OL1:0.1 | all | 1/712 | 0.058088 |
| 14M:A1-H-OL1:0.1 | early_1_142 | 1/142 | 0.046296 |
| 14M:A1-H-OL1:0.1 | late_570_712 | 0/143 | 0.088047 |
| 14M:A1-H-OL1:0.5 | all | 2/712 | 0.358326 |
| 14M:A1-H-OL1:0.5 | early_1_142 | 2/142 | 0.152127 |
| 14M:A1-H-OL1:0.5 | late_570_712 | 0/143 | 0.469601 |
| 14M:A1-H-OL1:1.0 | all | 76/712 | 0.802855 |
| 14M:A1-H-OL1:1.0 | early_1_142 | 2/142 | 0.232727 |
| 14M:A1-H-OL1:1.0 | late_570_712 | 74/143 | 1.000000 |
| 14M:A4-OL1:0.0 | all | 3/712 | 0.423954 |
| 14M:A4-OL1:0.0 | early_1_142 | 3/142 | 0.268591 |
| 14M:A4-OL1:0.0 | late_570_712 | 0/143 | 0.708087 |
| 14M:A4-OL1:0.01 | all | 7/712 | 0.328265 |
| 14M:A4-OL1:0.01 | early_1_142 | 4/142 | 0.292537 |
| 14M:A4-OL1:0.01 | late_570_712 | 3/143 | 0.826389 |
| 14M:A4-OL1:0.05 | all | 5/712 | 0.274579 |
| 14M:A4-OL1:0.05 | early_1_142 | 5/142 | 0.227409 |
| 14M:A4-OL1:0.05 | late_570_712 | 0/143 | 0.355437 |
| 14M:A4-OL1:0.1 | all | 2/712 | 0.188193 |
| 14M:A4-OL1:0.1 | early_1_142 | 2/142 | 0.237067 |
| 14M:A4-OL1:0.1 | late_570_712 | 0/143 | 0.189654 |
| 14M:A4-OL1:0.5 | all | 1/712 | 0.110045 |
| 14M:A4-OL1:0.5 | early_1_142 | 1/142 | 0.191674 |
| 14M:A4-OL1:0.5 | late_570_712 | 0/143 | 0.109974 |
| 14M:A7-OL1:0.0 | all | 712/712 | 1.000000 |
| 14M:A7-OL1:0.0 | early_1_142 | 142/142 | 1.000000 |
| 14M:A7-OL1:0.0 | late_570_712 | 143/143 | 1.000000 |
| 14M:A7-OL1:0.01 | all | 712/712 | 1.000000 |
| 14M:A7-OL1:0.01 | early_1_142 | 142/142 | 1.000000 |
| 14M:A7-OL1:0.01 | late_570_712 | 143/143 | 1.000000 |
| 14M:A7-OL1:0.05 | all | 712/712 | 1.000000 |
| 14M:A7-OL1:0.05 | early_1_142 | 142/142 | 1.000000 |
| 14M:A7-OL1:0.05 | late_570_712 | 143/143 | 1.000000 |
| 14M:A7-OL1:0.1 | all | 712/712 | 1.000000 |
| 14M:A7-OL1:0.1 | early_1_142 | 142/142 | 1.000000 |
| 14M:A7-OL1:0.1 | late_570_712 | 143/143 | 1.000000 |
| 14M:A7-OL1:0.5 | all | 605/712 | 1.000000 |
| 14M:A7-OL1:0.5 | early_1_142 | 142/142 | 1.000000 |
| 14M:A7-OL1:0.5 | late_570_712 | 40/143 | 0.818756 |
| 70M:A4-OL1:0.0 | all | 16/712 | 0.308660 |
| 70M:A4-OL1:0.0 | early_1_142 | 16/142 | 0.209093 |
| 70M:A4-OL1:0.0 | late_570_712 | 0/143 | 0.532868 |
| 70M:A4-OL1:0.01 | all | 18/712 | 0.246370 |
| 70M:A4-OL1:0.01 | early_1_142 | 18/142 | 0.228381 |
| 70M:A4-OL1:0.01 | late_570_712 | 0/143 | 0.542513 |
| 70M:A4-OL1:0.05 | all | 17/712 | 0.246433 |
| 70M:A4-OL1:0.05 | early_1_142 | 17/142 | 0.230524 |
| 70M:A4-OL1:0.05 | late_570_712 | 0/143 | 0.445780 |
| 70M:A4-OL1:0.1 | all | 19/712 | 0.286932 |
| 70M:A4-OL1:0.1 | early_1_142 | 19/142 | 0.236397 |
| 70M:A4-OL1:0.1 | late_570_712 | 0/143 | 0.423765 |
| 70M:A4-OL1:0.5 | all | 2/712 | 0.122281 |
| 70M:A4-OL1:0.5 | early_1_142 | 2/142 | 0.199121 |
| 70M:A4-OL1:0.5 | late_570_712 | 0/143 | 0.124565 |
| 70M:A7-OL1:0.0 | all | 712/712 | 1.000000 |
| 70M:A7-OL1:0.0 | early_1_142 | 142/142 | 1.000000 |
| 70M:A7-OL1:0.0 | late_570_712 | 143/143 | 1.000000 |
| 70M:A7-OL1:0.01 | all | 712/712 | 1.000000 |
| 70M:A7-OL1:0.01 | early_1_142 | 142/142 | 1.000000 |
| 70M:A7-OL1:0.01 | late_570_712 | 143/143 | 1.000000 |
| 70M:A7-OL1:0.05 | all | 712/712 | 1.000000 |
| 70M:A7-OL1:0.05 | early_1_142 | 142/142 | 1.000000 |
| 70M:A7-OL1:0.05 | late_570_712 | 143/143 | 1.000000 |
| 70M:A7-OL1:0.1 | all | 712/712 | 1.000000 |
| 70M:A7-OL1:0.1 | early_1_142 | 142/142 | 1.000000 |
| 70M:A7-OL1:0.1 | late_570_712 | 143/143 | 1.000000 |
| 70M:A7-OL1:0.5 | all | 712/712 | 1.000000 |
| 70M:A7-OL1:0.5 | early_1_142 | 142/142 | 1.000000 |
| 70M:A7-OL1:0.5 | late_570_712 | 143/143 | 1.000000 |
| 410M:A4-OL1:0.0 | all | 16/712 | 0.064253 |
| 410M:A4-OL1:0.0 | early_1_142 | 16/142 | 0.293952 |
| 410M:A4-OL1:0.0 | late_570_712 | 0/143 | 0.069839 |
| 410M:A4-OL1:0.01 | all | 17/712 | 0.062706 |
| 410M:A4-OL1:0.01 | early_1_142 | 17/142 | 0.258798 |
| 410M:A4-OL1:0.01 | late_570_712 | 0/143 | 0.068514 |
| 410M:A4-OL1:0.05 | all | 16/712 | 0.055255 |
| 410M:A4-OL1:0.05 | early_1_142 | 16/142 | 0.322847 |
| 410M:A4-OL1:0.05 | late_570_712 | 0/143 | 0.057933 |
| 410M:A4-OL1:0.1 | all | 18/712 | 0.058628 |
| 410M:A4-OL1:0.1 | early_1_142 | 18/142 | 0.353211 |
| 410M:A4-OL1:0.1 | late_570_712 | 0/143 | 0.066802 |
| 410M:A4-OL1:0.5 | all | 12/712 | 0.162521 |
| 410M:A4-OL1:0.5 | early_1_142 | 12/142 | 0.292116 |
| 410M:A4-OL1:0.5 | late_570_712 | 0/143 | 0.164849 |
| 410M:A7-OL1:0.0 | all | 4/712 | 0.105100 |
| 410M:A7-OL1:0.0 | early_1_142 | 4/142 | 0.153150 |
| 410M:A7-OL1:0.0 | late_570_712 | 0/143 | 0.148967 |
| 410M:A7-OL1:0.01 | all | 5/712 | 0.093108 |
| 410M:A7-OL1:0.01 | early_1_142 | 5/142 | 0.147546 |
| 410M:A7-OL1:0.01 | late_570_712 | 0/143 | 0.134882 |
| 410M:A7-OL1:0.05 | all | 4/712 | 0.074688 |
| 410M:A7-OL1:0.05 | early_1_142 | 4/142 | 0.157804 |
| 410M:A7-OL1:0.05 | late_570_712 | 0/143 | 0.104900 |
| 410M:A7-OL1:0.1 | all | 6/712 | 0.061264 |
| 410M:A7-OL1:0.1 | early_1_142 | 6/142 | 0.164939 |
| 410M:A7-OL1:0.1 | late_570_712 | 0/143 | 0.088897 |
| 410M:A7-OL1:0.5 | all | 3/712 | 0.196496 |
| 410M:A7-OL1:0.5 | early_1_142 | 3/142 | 0.211122 |
| 410M:A7-OL1:0.5 | late_570_712 | 0/143 | 0.201571 |
