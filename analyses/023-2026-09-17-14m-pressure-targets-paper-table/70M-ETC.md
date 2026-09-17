# Provisional 70M h-only OL1 ETC

Match Run 018 canonical seed-1234 initialization, realized data order, FP16, microbatch 4 / accumulation 256, 712 updates, 1,493,172,224 tokens, AdamW recipe, complete validation/diagnostics; change pressure sites to h only at lambda=1,b=1.

Timing basis: one H200 per independent condition. These are estimates, not a new calibration or a launch definition.

| Condition | kappa | Worker time, including evaluation/diagnostics/checkpointing |
|---|---:|---:|
| A4-OL1(h) | 0.05 | 90–105 min |
| A4-OL1(h) | 0.5 | 90–105 min |
| A7-OL1(h) | 0.05 | 95–115 min |
| A7-OL1(h) | 0.5 | 95–115 min |

Add 20–45 minutes once to the elapsed schedule for provision, setup, short calibration, transfer and local verification, assuming available capacity and functioning transfers.

| Schedule | Total elapsed estimate |
|---|---:|
| Four H200s, four conditions in parallel | 115–160 min (1.92–2.67 h) |
| Two H200s, two waves | 205–265 min (3.42–4.42 h) |
| One H200, four conditions sequentially | 390–485 min (6.50–8.08 h) |

## Evidence and uncertainty

The exact Run 018 A7 preflight measured five complete boundaries: median 7.798 s, range 7.792–7.932 s. Completed all-site runs provide the stronger total-worker-time evidence below.

| Historical all-site condition | Worker min | Peak reserved GiB |
|---|---:|---:|
| a4-ol1-kappa-0 | 90.13 | 9.986 |
| a4-ol1-kappa-0p01 | 91.52 | 9.986 |
| a4-ol1-kappa-0p05 | 159.24 | 9.986 |
| a4-ol1-kappa-0p1 | 155.08 | 9.986 |
| a4-ol1-kappa-0p5 | 88.67 | 9.986 |
| a7-ol1-kappa-0 | 97.29 | 10.174 |
| a7-ol1-kappa-0p01 | 96.21 | 10.174 |
| a7-ol1-kappa-0p05 | 150.51 | 10.174 |
| a7-ol1-kappa-0p1 | 98.48 | 10.174 |
| a7-ol1-kappa-0p5 | 98.85 | 10.174 |

Typical all-site H200 workers took 88.67–98.85 minutes. Three slower workers took 150.51–159.24 minutes; the contemporaneous Run 018 recovery notes identify regional contention. Threshold alone is not a runtime predictor for these dense kernels.

The h-only estimate conservatively reuses the all-site timings without assuming a speed gain, and adds room for a larger retained checkpoint inventory. The exact h-only path still needs a short calibration before a launch ETC can be claimed.

Checkpoint storage is approximately 5.07 GB per condition (about 20.3 GB total) for the Run 032-style inventory, plus small diagnostics/logs. Run 018 retained only the final recovery checkpoint (about 0.85 GB per condition). Input cache and initialization are about 6.25 GB per worker. The 20–45 minute transfer allowance is provisional, not a measured guarantee for a new host/network.

Observed all-site memory was 9.99–10.17 GiB. This does not establish the h-only workload's memory headroom or runtime on the local 12 GB laptop. No valid matched 70M A100 training timing was found: the historical A100 attempt stopped at initialization. H200 ETC must not be relabeled as A100 ETC.

No prices or capacity are quoted, and no new run or cloud resource is created. Reproduce with `python analyses/023-2026-09-17-14m-pressure-targets-paper-table/02_estimate_70m.py`; exact inputs/hashes are in `70m-etc.json`.
