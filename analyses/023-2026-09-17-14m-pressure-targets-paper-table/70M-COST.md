# 70M h-only OL1 — provisional RunPod cost

Quote checked on 17 September 2026 at 11:28 UTC. Scope: four independent
conditions, A4-OL1(h) and A7-OL1(h), each at kappa 0.05 and 0.5.
Elapsed-time assumptions are retained in [70M-ETC.md](70M-ETC.md).

RunPod's [current public pricing](https://www.runpod.io/pricing), updated
13 September 2026, lists H200 Pods at **USD4.59 per GPU-hour**. The current
[billing documentation](https://docs.runpod.io/accounts-billing/billing)
lists running container and volume disks at USD0.10/GB/month and no
data-transfer fees. Compute continues billing during setup and retrieval.
The MCP catalog request failed at the transport layer; the public pricing
and billing pages are the successful price sources. Four-Pod availability
was not established by this quote.

| Schedule | Estimated elapsed time | GPU-hours, conservatively billed for the full schedule | Compute cost |
|---|---:|---:|---:|
| Four H200s in parallel | 115–160 min | 7.667–10.667 | USD35.19–48.96 |
| Two H200s, two waves | 205–265 min | 6.833–8.833 | USD31.37–40.55 |
| One H200, sequential | 390–485 min | 6.500–8.083 | USD29.84–37.10 |

Arithmetic: `GPU count * elapsed minutes / 60 * 4.59`. Each elapsed range
already includes the 20–45 minute provision/setup/preflight/retrieval/verification
allowance from the ETC note; it is not added a second time. Releasing a
finished worker after verified retrieval can reduce the conservative total.
Fewer workers avoid duplicating some billed setup time; the training budget
per condition is unchanged.

Allowing 80 GB of temporary running disk per Pod adds at most about USD0.12
within these ranges (`GB * 0.10 / 720 * Pod-hours`, approximate 30-day month).
Rounded incremental totals are therefore **USD35–50 with four H200s**,
**USD31–41 with two**, or **USD30–38 sequentially**. These are service-usage
estimates in USD before any applicable tax or currency conversion. The
pre-existing network volume's continuing charge is separate.

For four parallel workers, a **USD65–70 planning allowance** covers about
3.5–3.8 hours at the quoted rate plus these small disk costs. This provides
room for the prior cohort's slower workers and retrieval, but is not a
guaranteed maximum or an installed spending guard. Further delays continue
accruing charges. No new resource or launch authorization is created by
this cost estimate.
