# Table 2: Threshold placement at fixed h-only pressure

Seven-site minus four-site, with OL1(h) held fixed. One final checkpoint per trained setting.

| kappa | 14M: ΔS (pp) | 14M: ΔL (nats/token) | 70M: ΔS (pp) | 70M: ΔL (nats/token) |
|---|---:|---:|---:|---:|
| 0 | -0.0100 | +0.0041 | -0.0056 | +0.0443 |
| 0.01 | +0.2703 | -0.0064 | +0.0382 | -0.0063 |
| 0.05 | +1.0904 | -0.0006 | +0.3283 | +0.0707 |
| 0.1 | +1.5601 | +0.0087 | +0.5611 | +0.0169 |
| 0.5 | +6.4362 | +0.0094 | +2.9967 | +0.0315 |

Sources and exact checkpoint pairs: `data/paper-derived.json`. Positive loss changes are worse; positive sparsity changes mean more logical zero products. These five settings are not independent seeds.
