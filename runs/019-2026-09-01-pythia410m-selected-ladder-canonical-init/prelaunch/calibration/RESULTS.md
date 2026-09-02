# Run 019 GPU calibration results and launch options

## Scope

These are non-evidence measurements used only to select infrastructure for the
twelve approved 410M conditions. Both candidates ran the same five exact
boundaries for A0, A4-OL1 at `kappa=0.5`, and A7-OL1 at `kappa=0.5`; the first
boundary was warm-up. Complete validation, diagnostics, checkpoint operations,
and control TEAL work are included in the projection. Provisioning, package
installation, input delivery, and result retrieval are excluded.

The comparator verified identical initialization, schedule, and run-code
identities and left `selection` null. A1-H uses the measured A0 control timing.
All A4 and A7 thresholds use their topology's measured `kappa=0.5` path; this is
conservative if smaller thresholds run faster.

## Same-SKU twelve-way launch

| Option | Price/GPU-h | Fleet burn | Longest condition / makespan | Aggregate GPU-h | Projected compute cost | Peak reserved | Free VRAM fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Secure A40 | $0.44 | $5.28/h | 47.91 h | 492.65 | $216.76 | 34.79 GiB | 21.70% |
| Secure A100 SXM 80 GB | $1.59 | $19.08/h | 19.46 h | 203.50 | $323.56 | 34.73 GiB | 56.18% |
| Community H200 SXM | $3.59 | $43.08/h | 12.19 h | 128.27 | $460.49 | 34.79 GiB | 75.20% |
| Secure H200 SXM | $4.59 | $55.08/h | 12.19 h | 128.27 | $588.76 | 34.79 GiB | 75.20% |

The A100 option is 28.45 hours faster (59.4% less wall time) and costs $106.79
more (49.3% more compute). Equivalently, the speed premium is about $3.75 for
each makespan hour saved. Both candidates pass the configured 10% VRAM-headroom
gate; A100 has substantially more failure and kernel-workspace margin.

Per-condition group projections are:

| Conditions | Count | A40 ETC/cost each | A100 ETC/cost each | H200 ETC each | H200 Community/Secure cost each |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0, A1-H (including TEAL) | 2 | 21.04 h / $9.26 | 9.24 h / $14.70 | 5.861 h | $21.04 / $26.90 |
| A4-OL1, all five `kappa` | 5 | 42.20 h / $18.57 | 17.54 h / $27.89 | 11.117 h | $39.91 / $51.03 |
| A7-OL1, all five `kappa` | 5 | 47.91 h / $21.08 | 19.46 h / $30.94 | 12.192 h | $43.78 / $55.96 |

Condition-specific hard-guard proposals, including roughly 20% runtime margin
plus setup/retrieval time, are 28/52/60 hours for A40 A0/A4/A7 and 13/23/26
hours for A100 A0/A4/A7. The corresponding fleet-wide maximum compute
envelopes are approximately $271.04 for A40 and $430.89 for A100, before any
storage or transfer charge. These are guards, not expected bills.

The later user-approved fastest-GPU calibration measured H200 directly. Its
per-condition projections are 5.861 hours for A0/A1-H, 11.117 hours for each
A4-OL1 condition, and 12.192 hours for each A7-OL1 condition. Applying the
approved `1.75 x measured projection + 0.5 h measured setup/transfer` policy
gives condition guards of 10.8, 20.0, and 21.9 hours. H200 passed every
finite-boundary, initialization-identity, diagnostic, and VRAM-headroom check.
Its calibration JSON SHA-256 is
`535796bcb9d12a2cc3b95955128c1382b1af175ce7e88941f3b724886153bb8e`.

## Capacity-constrained waves

The following exact schedules use the projected condition durations on an
identical-SKU fleet. Aggregate compute cost is unchanged in the ideal case;
fewer Pods lower instantaneous burn and may reuse package/input setup, but
increase wall time.

| SKU | GPUs | Scheduled makespan | Fleet burn | Projected compute cost |
| --- | ---: | ---: | ---: | ---: |
| A40 | 12 | 47.91 h | $5.28/h | $216.76 |
| A40 | 8 | 84.40 h | $3.52/h | $216.76 |
| A40 | 6 | 90.11 h | $2.64/h | $216.76 |
| A40 | 4 | 132.31 h | $1.76/h | $216.76 |
| A100 SXM | 12 | 19.46 h | $19.08/h | $323.56 |
| A100 SXM | 8 | 35.08 h | $12.72/h | $323.56 |
| A100 SXM | 6 | 37.00 h | $9.54/h | $323.56 |
| A100 SXM | 4 | 54.54 h | $6.36/h | $323.56 |
| H200 Community | 12 | 12.19 h | $43.08/h | $460.49 |
| H200 Community | 8 | 22.23 h | $28.72/h | $460.49 |
| H200 Community | 6 | 23.31 h | $21.54/h | $460.49 |
| H200 Community | 4 | 34.43 h | $14.36/h | $460.49 |

Eight GPUs offer little advantage over six because five A7 and five A4 jobs
still force a second wave. Twelve GPUs are the meaningful fast-launch point.

## Decision interpretation

- Fastest measured matched design: twelve H200 SXM Pods, approximately 12.19
  hours and $460.49--$588.76 compute depending on Community/Secure capacity.
- Cheapest matched design: twelve A40 Pods, approximately 47.91 hours and
  $216.76 compute.
- If capacity prevents twelve Pods, six same-SKU Pods are the efficient wave
  fallback; eight gives only a small makespan improvement.

A mixed-SKU fleet can interpolate cost and time arithmetically, but it is not
the prepared design: hardware-dependent kernels would vary systematically by
condition. For reference only, putting A0/A1-H on A40 and A4/A7 on A100 would
project to 21.04 hours and $312.68; putting A0/A1-H/A4 on A40 and A7 on A100
would project to 42.20 hours and $266.07. Either requires explicit design
approval and a config change. A same-SKU cohort is preferred.

At the 2026-09-02 closeout refresh, Secure A40 was $0.44/h with high aggregate
stock (EU-SE-1 and CA-MTL-1), while Secure A100 SXM was $1.59/h with medium
aggregate stock; A100 stock at the existing EUR-IS-1 volume's datacenter was
low. The uncalibrated Community RTX A6000 remained $0.33/h with low stock. It is
not an admissible measured option without another calibration.

At the final H200 refresh, H200 SXM stock was Low across seven listed data
centers. Community was $3.59/h and Secure was $4.59/h. The human selected H200
and approved a twelve-way fastest launch, with Community preferred and Secure
allowed as a same-SKU capacity fallback. The $401.53 post-calibration balance
did not cover even the $460.49 all-Community central projection, so launch is
blocked only until the funded balance covers the selected mix plus variance.

## Full-run staging recommendation

Do not repeat twelve laptop uploads. Calibration delivery took 12m39s to A100
and 1h03m50s to A40 for the same 7.070 GiB payload. Pre-stage the single
hash-pinned payload in an approved object store, then let each isolated 60 GB
Pod volume download it concurrently and verify the existing manifest. The
pre-existing EUR-IS-1 network volume exposes an S3-compatible path and is a
possible staging source only after explicit approval; it was not touched here.

Keep one condition per isolated Pod workspace. A shared mounted workspace would
require unique venv, control, attempt, and checkpoint paths and careful
single-writer behavior. The measured final checkpoint is 4,864,312,106 bytes,
so the twelve retained final checkpoints alone total about 54.36 GiB. Retrieve
and hash-verify each condition as it finishes, then delete that Pod rather than
waiting for the slowest condition.

The H200 SKU, twelve-way concurrency, and launch are human-approved. Source and
execution configuration are being amended and tested; the remaining external
gate is adequate funded balance for the live Community/Secure capacity mix.
