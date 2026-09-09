# Additional evidence: conditional requests, not launched

No new runs are needed for the wording in this revision. The requests below
are outside EXISTING_EVIDENCE authorization and are not experiment designs
or launch approvals. No numbered run folder or experiment code was created.

- E01 (optional, medium effort): if exact achieved-sparsity matching becomes
  central, evaluate a few additional calibration targets on the fixed 410M A0
  and A1-H checkpoints. Existing informative points differ by 1.521 sparsity pp;
  the paper states that mismatch. Use the same training-only calibration and
  all 338 validation blocks; stop after the predeclared target list, retaining
  every result. Lower total loss at closer sparsity would strengthen the local
  comparison; crossing would narrow it. No additional training is needed.
- E02: the missing-log trigger is not applicable: all 24,208 OL1 boundary records
  exist. The fixed-state geometry and observed cap fractions need no new run.
  A trajectory-robustness claim would instead require a separately designed
  multi-weight study. A global rescaling that stays saturated is uninformative.
- E03 (optional, medium effort per run): confirm the central 14M high-threshold
  A4-OL1/A7-OL1 pair with at least one additional matched seed pair (two full
  training conditions). Choose seed count before launch; one extra pair remains
  weak evidence for variability. Retain the same 712-update protocol and full
  validation, and stop at that budget. A reversal must narrow the claim, not be
  dropped. More seeds would strengthen estimates but increase cost.
- E04 (optional, high effort): jointly test longer training for 410M A0 and a
  selected high-threshold sparse recipe, preserving optimizer state if the
  question is continuation. Establish token budget, LR treatment, checkpoint
  retention and validation cadence before execution; stop at the approved
  budget. Improved loss alone does not establish restoration of 70M/410M ordering
  or shrinking sparse penalty. Dense/sparse responses must both be reported.

Expected monetary cost is not quoted without a confirmed workload and live
hardware quote. Exact duration, cost ceiling, storage and transfer envelope belong
to the design/launch confirmation required by AGENTS.md. The current task does
not need those paid actions. Independent seeds are the clearest next evidence
for the central contrast; continuation addresses a different causal question.
