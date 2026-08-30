# Gradient-interaction tables

Each condition contains 712 complete optimizer boundaries. Conflict fractions
count boundaries before dividing; quantiles are over boundary-level scalar
diagnostics. The task and pressure gradients in the first two tables are
separate, unweighted gradients computed before naive L1 combines and globally
clips them.

## Naive-L1 raw task-pressure interaction

| lambda | negative dot (n/712) | negative dot (%) | cosine q05 | cosine q25 | median cosine | cosine q75 | cosine q95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 350/712 | 49.16 | -0.269814 | -0.104416 | +0.002505 | +0.089236 | +0.278954 |
| 0.1 | 387/712 | 54.35 | -0.284224 | -0.133045 | -0.017370 | +0.094996 | +0.293880 |
| 0.5 | 454/712 | 63.76 | -0.391952 | -0.207196 | -0.065280 | +0.069323 | +0.303188 |
| 1 | 505/712 | 70.93 | -0.479775 | -0.290843 | -0.105177 | +0.037859 | +0.271282 |

## Naive-L1 combined raw-gradient alignment

For a boundary with a negative task-pressure dot, the alignment offset is
`-lambda * <g_task,g_pressure> / ||g_task||^2`. Thus
`<g_task,g_task + lambda*g_pressure> = ||g_task||^2 * (1 - offset)`.
The offset describes the raw pre-clip direction; it is not a realized loss
change or an AdamW-step measurement.

| lambda | combined raw dot < 0 (n/712) | negative-pressure boundaries | median alignment offset (%) | q95 offset (%) | maximum offset (%) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0/712 | 350 | 0.169 | 0.496 | 0.778 |
| 0.1 | 0/712 | 387 | 0.354 | 0.925 | 1.839 |
| 0.5 | 0/712 | 454 | 1.504 | 4.212 | 6.159 |
| 1 | 0/712 | 505 | 2.675 | 9.929 | 14.110 |

## OL1 AdamW-relative interaction and trust cap

The adaptive quantities compare `d_task = m_hat/D` with
`d_pressure = g_pressure/D`, where `D = sqrt(v_hat) + adam_eps` uses
task-only AdamW state. A negative dot triggers projection. The residual
post-projection cosine is reported only for projected boundaries.

| lambda | negative adaptive dot / projected (n/712) | cosine-before q05 | median cosine before | cosine-before q95 | max abs projected cosine after | trust cap active (n/712) | median raw ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 697/712 | -0.046482 | -0.023622 | -0.004112 | 7.885e-09 | 0/712 | 0.029871 |
| 0.1 | 711/712 | -0.065849 | -0.029460 | -0.011470 | 1.126e-08 | 1/712 | 0.058088 |
| 0.5 | 711/712 | -0.120903 | -0.047163 | -0.026382 | 2.395e-08 | 2/712 | 0.358326 |
| 1 | 711/712 | -0.125649 | -0.058347 | -0.037086 | 1.900e-08 | 76/712 | 0.802855 |

All eight conditions begin from the same initialization and have
first-boundary raw cosines within 1.800e-08.
Because lambda does not weight the recorded component gradients, later
differences in conflict describe lambda-dependent training trajectories,
not an algebraic change in angle from multiplying by lambda.

The OL1 projection constrains alignment with the task-only adaptive
direction. It does not guarantee non-increase of the current task loss or
validation loss, and the trust budget bounds rather than eliminates
hyperparameter sensitivity.
