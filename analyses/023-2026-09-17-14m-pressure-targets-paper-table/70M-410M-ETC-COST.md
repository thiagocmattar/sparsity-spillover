# 70M and 410M h-only OL1: two versus five thresholds

Planning resumed at the user's request on 17 September 2026. This estimate
creates no experiment, calibration, Pod, or launch authorization. The earlier
70M-only estimates and freeze record remain historical records.

Both grids include **A4-OL1(h) and A7-OL1(h)**, with pressure only at h,
lambda=1 and b=1. A4 gates a,m,h,z; A7 additionally gates q_post,k_post,v.
The two-threshold grid is {0.05,0.5}: four conditions per scale, eight total.
The full grid is {0,0.01,0.05,0.1,0.5}: ten per scale, twenty total.

The estimate matches each scale's previous canonical Runs 018/019 initialization,
seed 1234, realized data order, FP16 AdamW recipe, microbatch 4 with accumulation 256,
712 updates and 1,493,172,224 training input tokens. It includes the previous
diagnostics and full final-checkpoint evaluation: 500 validation documents,
338 complete 2048-token blocks, excluded tail 1444. There are no extra seeds,
old-condition reruns, or kernel benchmarks in these totals.

## Parallel schedule and expected usage

One independent GPU per condition, with both scales started concurrently.
The 70M workers are released after their own artifacts are retrieved and
verified; they are not held until 410M finishes. Within each scale the cost
calculation conservatively holds every worker through its slowest worker's
estimated completion plus setup/retrieval overhead. Early individual release
can reduce the bill. Ranges are planning judgments, not confidence intervals.

| Model and hardware | Two kappas: GPUs | Elapsed h | Incremental USD | All five: GPUs | Elapsed h | Incremental USD |
|---|---:|---:|---:|---:|---:|---:|
| 70M, H200 | 4 | 1.92-2.67 | 28-50 | 10 | 2.08-3.42 | 75-160 |
| 410M, A100 SXM 80GB | 4 | 19.5-23 | 110-150 | 10 | 20-24 | 280-390 |
| 410M, H200 alternative | 4 | 13-15.5 | 188-286 | 10 | 13.5-16.5 | 487-761 |

| Combined option | Two kappas: elapsed h / USD | All five: elapsed h / USD |
|---|---:|---:|
| H200 for 70M + A100 for 410M | 19.5-23 / **137-197** | 20-24 / **356-543** |
| H200 for both scales | 13-15.5 / **215-335** | 13.5-16.5 / **562-918** |

Combined rows use the unrounded calculation rather than summing rounded cells.
There are eight concurrent GPUs for the two-threshold grid, twenty for the full
grid. These are conditional schedules, not statements that that many GPUs are
available. Fewer workers increase elapsed time; compute per condition is broadly
unchanged, while some setup can be reused.

Among these two well-supported options, the lower-cost choice is H200 for 70M
and A100 for 410M. A practical planning
reserve is **USD250 for two kappas** or **USD700 for all five** (roughly 25-30%
above the upper normal estimate). These are funding allowances, not installed
spending caps, approval requests, or guaranteed maximum bills.

## Price and capacity evidence

The authenticated RunPod catalog was saved at **2026-09-17 13:04:17 UTC** in
[70m-410m-price-snapshot.json](70m-410m-price-snapshot.json).

| GPU | Community USD/GPU-h | Secure USD/GPU-h | Aggregate stock |
|---|---:|---:|---|
| A100 SXM 80GB | 1.39 | 1.59 | Low |
| H200 SXM | 3.59 | 4.59 | Low |

The low cost endpoint uses Community and the high endpoint Secure. Catalog
stock is not a worker count or a reservation. The account's read-only
`runpodctl user` response reports an USD80/hour spend limit. Twenty Secure
H200s would cost USD91.80/hour before disk, so the fully concurrent all-Secure
H200 case is **not feasible under the present account limit**. Running ten
H200s for 70M and then ten for 410M instead gives about **15.6-19.9 hours**
combined, with approximately the same conservative per-scale bill. A Community
fleet or permitted mixed-tier allocation changes the burn; it still needs
real capacity confirmation. The combined H200/A100 option is below the current
limit even with both scales on Secure (USD61.80/hour before disk at twenty GPUs).

Costs include a total running disk allowance of 80 GB per 70M Pod and 120 GB per 410M
Pod, at USD0.10/GB/month, prorated with a 30-day month. RunPod states there are no
ingress/egress fees; compute continues while transfers run. See the current
[RunPod Pod pricing documentation](https://docs.runpod.io/pods/pricing).
Taxes, currency conversion, third-party staging charges and the pre-existing
network volume's continuing cost are excluded.

## Timing evidence and limits

Run 018 completed all-site 70M H200 workers in 88.67-98.85 minutes normally;
three took 150.51-159.24 minutes under recorded contention. The existing
[70M estimate](70M-ETC.md) uses 90-105 minutes for A4 and 95-115 for A7.
No valid matched 70M A100 training timing exists in the retained evidence.

Run 019's completed 410M A100 workers took 17.50-17.55 hours for A4 and
18.46-19.53 hours for A7. Its H200 calibration projected 11.117 hours for A4
and 12.192 for A7, using four measured complete boundaries after one warmup,
plus complete validation/diagnostic/checkpoint operations. These H200 values
are projections, not completed H200 410M workers. The new planning ranges are:

| Worker, including scientific evaluation/diagnostics | A4 hours | A7 hours |
|---|---:|---:|
| 70M H200 | 1.50-1.75 | 1.58-1.92 |
| 410M A100 | 17.5-19 | 18.5-21 |
| 410M H200 | 11-12 | 12-13.5 |

The historical pressure scope is all sites. **No speed gain from h-only OL1 is
assumed.** These measurements establish a conservative basis, not an exact
h-only calibration. A new launch ETC must measure the actual h-only path.
There is no reliable threshold-to-runtime adjustment for these dense training
kernels, so all kappas use their topology's same range.

Normal provision, preflight, extra checkpoint handling, retrieval and local
hash verification allowances added to the slowest worker are:

| Scale | Two kappas | All five |
|---|---:|---:|
| 70M | 20-45 minutes | 30-90 minutes |
| 410M | 1-2 hours | 1.5-3 hours |

The full-grid allowance is larger than the earlier four-condition 70M estimate
because more checkpoint bytes must be retrieved. It assumes a hash-verified
cache staged once and distributed within the cloud, and at least **50 MB/s
aggregate output download throughput**, with transfers overlapped where possible.
These are explicit planning assumptions, not measurements on future hosts.
Slow laptop uploads repeated for each Pod, congestion, capacity queues or
recovery retries can materially exceed both ETC and cost. Every extra held
GPU-hour adds the quoted GPU rate plus disk; for example, keeping all ten 410M
A100s for one extra hour adds USD13.90-15.90 before disk.

## Retention and reproducibility

The storage estimate carries forward the recent 14M retention pattern:
12 FP32 model snapshots at steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712, plus
optimizer/scaler/RNG at 256, 512, 712. Two optimizer tensors per parameter give
approximately 18 model-size equivalents, excluding small metadata. This is a
planning assumption for later design confirmation; it is larger than the
final-only recovery inventory in Runs 018/019.

| Scale | Approximate GB/condition | Two-kappa grid GB | Full grid GB |
|---|---:|---:|---:|
| 70M | 5.07 | 20.28 | 50.71 |
| 410M | 29.18 | 116.74 | 291.84 |
| Both | -- | **137.02** | **342.55** |

Logs/diagnostics are additional small artifacts. Input cache plus canonical
initialization is approximately 6.25 GB per 70M worker and 7.59 GB per 410M worker.
Final checkpoints and all agreed artifacts must be copied and hash-verified
before Pod deletion. Changing retention would change the transfer allowance;
no retention reduction is assumed here.

Reproduce the arithmetic without any cloud action:

```powershell
.venv/Scripts/python.exe analyses/023-2026-09-17-14m-pressure-targets-paper-table/03_estimate_70m_410m.py
```

[70m-410m-etc-cost.json](70m-410m-etc-cost.json) retains full-precision ranges,
GPU-hours, tier-specific costs, historical worker rows, all input SHA-256
hashes, assumptions and exclusions. This estimate changes no scientific input
or result and does not select or approve either grid.

Verification: deterministic regeneration passed; all 24 recorded source hashes,
both grids, GPU-hour/cost arithmetic and local document links were checked.
No training code changed and no GPU work ran for this estimate.
