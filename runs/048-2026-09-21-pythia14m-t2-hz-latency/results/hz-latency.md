# Fixed-checkpoint h/z latency control

| Mode | h skip | z skip | Host latency (ms) | Process range (ms) |
| --- | --- | --- | ---: | ---: |
| A | True | True | 0.558789 | 0.554875–0.561594 |
| B | False | True | 0.648529 | 0.644908–0.651165 |
| C | True | False | 0.562998 | 0.558981–0.567270 |
| D | False | False | 0.641538 | 0.636399–0.645749 |

Thresholds, gate values and zero masks are identical in A–D on all338 validation blocks.
Every final-process logit equals the common eager reference numerically (maximum absolute difference0).

- h_given_z: 89.740 us saved; 13.837% reduction relative to its disabled mode; 1.16060x.
- z_given_h: 4.208 us saved; 0.747% reduction relative to its disabled mode; 1.00753x.
- joint_hz: 82.749 us saved; 12.899% reduction relative to its disabled mode; 1.14809x.
- Interaction D−B−C+A: -11.199 us.

Ranges describe three fresh processes, not population confidence intervals. Both tile skipping and short-row scalar execution are part of the enabled sparse path.
