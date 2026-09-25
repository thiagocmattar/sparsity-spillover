# Conditional tile-bypass and short-row effects

14M T7/Pall, kappa=0.5; fixed checkpoint and gates. Three processes per mode.

| Execution | Full-model latency (ms) | Process range (ms) |
|---|---:|---:|
| Neither | 0.661507 | [0.660785, 0.661885] |
| Tile only | 0.494815 | [0.493861, 0.495391] |
| Short-row only | 0.579327 | [0.577867, 0.580738] |
| Both | 0.496476 | [0.495721, 0.496927] |
| Frozen K050 | 0.496453 | [0.495900, 0.497442] |

| Effect | Saved time (us) | Difference span (us) |
|---|---:|---:|
| Tile, given short rows | +82.851 | [+80.941, +85.017] |
| Short rows, given tile bypass | -1.661 | [-3.066, -0.331] |
| Joint | +165.031 | [+163.859, +166.164] |
| Tile alone | +166.691 | [+165.395, +168.025] |
| Short rows alone | +82.180 | [+80.047, +84.018] |
| Interaction | -83.841 | [-87.084, -80.378] |

Positive differences mean savings. Spans are process extrema, not confidence intervals.
Conditional tile plus conditional short-row savings equal joint savings plus interaction.
The dense fallback retains the original padded layout; it is not an optimal dense baseline.
Source: `mechanism-latency.json`, audited by `07_reduce.py`.
