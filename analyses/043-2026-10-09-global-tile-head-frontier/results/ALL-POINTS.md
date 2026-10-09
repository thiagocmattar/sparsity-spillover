# All retained execution coordinates

87 retained 14M executions, including native and skip-disabled controls, not 87 trained models.
Loss is the canonical final-checkpoint FP16 full-validation result; latency is
complete B1/T2048/full-logits forward on RTX 5090, using the declared timing precision.
All modes for a checkpoint share its quality coordinate. A native or no-skip mode
keeps the trained gates. The CSV records source paths, precision, gate shape,
execution cohort, process ranges where available, and stable point IDs.
λ appears only for the historical pressure-weight sweep; it is not a gate threshold.
T1 retains its historical ReLU intervention, rather than a κ threshold sweep.

## BF16 block-kernel cohort

| Model | Training recipe | Body gate κ or pressure λ | Head κ | Execution | Loss | Full host ms | Session | 14M frontier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | ReLU h / Ph | λ=0.05 | — | K050 | 5.198062 | 0.632565 | Run029 | no |
| 14M | ReLU h / Ph | λ=0.1 | — | K050 | 5.159391 | 0.633082 | Run029 | no |
| 14M | ReLU h / Ph | λ=0.5 | — | K050 | 5.110235 | 0.616329 | Run029 | yes |
| 14M | ReLU h / Ph | λ=1 | — | K050 | 5.121030 | 0.599598 | Run029 | yes |
| 14M | T4 scalar / P0 | 0 | — | K050 | 5.470497 | 0.633912 | Run029 | no |
| 14M | T4 scalar / P0 | 0.01 | — | K050 | 5.466500 | 0.633912 | Run029 | no |
| 14M | T4 scalar / P0 | 0.05 | — | K050 | 5.434110 | 0.629581 | Run029 | no |
| 14M | T4 scalar / P0 | 0.1 | — | K050 | 5.419642 | 0.592548 | Run029 | no |
| 14M | T4 scalar / P0 | 0.5 | — | K050 | 5.659680 | 0.478819 | Run029 | yes |
| 14M | T4 scalar / Pall | 0 | — | K050 | 5.458170 | 0.633282 | Run029 | no |
| 14M | T4 scalar / Pall | 0.01 | — | K050 | 5.458277 | 0.620135 | Run029 | no |
| 14M | T4 scalar / Pall | 0.05 | — | K050 | 5.489803 | 0.572679 | Run029 | no |
| 14M | T4 scalar / Pall | 0.1 | — | K050 | 5.548323 | 0.533708 | Run029 | no |
| 14M | T4 scalar / Pall | 0.5 | — | K050 | 6.037987 | 0.459476 | Run029 | yes |
| 14M | T7 scalar / P0 | 0 | — | K050 | 5.468401 | 0.633961 | Run029 | no |
| 14M | T7 scalar / P0 | 0.01 | — | K050 | 5.458822 | 0.638352 | Run029 | no |
| 14M | T7 scalar / P0 | 0.05 | — | K050 | 5.437888 | 0.635332 | Run029 | no |
| 14M | T7 scalar / P0 | 0.1 | — | K050 | 5.428681 | 0.594166 | Run029 | no |
| 14M | T7 scalar / P0 | 0.5 | — | K050 | 5.702923 | 0.479137 | Run029 | no |
| 14M | T7 scalar / Pall | 0 | — | K050 | 5.480184 | 0.634030 | Run029 | no |
| 14M | T7 scalar / Pall | 0.01 | — | K050 | 5.475797 | 0.638417 | Run029 | no |
| 14M | T7 scalar / Pall | 0.05 | — | K050 | 5.462811 | 0.634519 | Run029 | no |
| 14M | T7 scalar / Pall | 0.1 | — | K050 | 5.429497 | 0.592979 | Run029 | no |
| 14M | T7 scalar / Pall | 0.5 | — | K050 | 5.829390 | 0.473366 | Run029 | no |
| 14M | T4 scalar / Ph | 0 | — | K050 | 5.215749 | 0.603658 | Run029 | no |
| 14M | T4 scalar / Ph | 0.01 | — | K050 | 5.204504 | 0.592158 | Run029 | no |
| 14M | T4 scalar / Ph | 0.05 | — | K050 | 5.195590 | 0.561801 | Run029 | yes |
| 14M | T4 scalar / Ph | 0.1 | — | K050 | 5.228687 | 0.529309 | Run029 | yes |
| 14M | T4 scalar / Ph | 0.5 | — | K050 | 5.722666 | 0.462252 | Run029 | yes |
| 14M | T7 scalar / Ph | 0 | — | K050 | 5.219827 | 0.617605 | Run033 | no |
| 14M | T7 scalar / Ph | 0.01 | — | K050 | 5.198102 | 0.603708 | Run033 | no |
| 14M | T7 scalar / Ph | 0.05 | — | K050 | 5.194959 | 0.574876 | Run033 | no |
| 14M | T7 scalar / Ph | 0.1 | — | K050 | 5.237395 | 0.541844 | Run033 | no |
| 14M | T7 scalar / Ph | 0.5 | — | K050 | 5.732049 | 0.473888 | Run033 | no |
| 14M | T2 scalar / Ph | 0 | — | K050 | 5.128602 | 0.612348 | Run041 | no |
| 14M | T2 scalar / Ph | 0.01 | — | K050 | 5.130649 | 0.608618 | Run041 | no |
| 14M | T2 scalar / Ph | 0.05 | — | K050 | 5.134379 | 0.590124 | Run041 | yes |
| 14M | T2 scalar / Ph | 0.1 | — | K050 | 5.151098 | 0.573427 | Run041 | yes |
| 14M | T2 scalar / Ph | 0.5 | — | K050 | 5.536255 | 0.506323 | Run044 | yes |
| 14M | Base | — | — | kernel | 5.208583 | 0.651573 | Run029 | no |
| 14M | Base | — | — | PyTorch | 5.208583 | 0.655442 | Run029 | no |
| 14M | T2 tile / Ph | 0 | — | sparse | 5.553770 | 0.540220 | Run061 | no |
| 14M | T2 tile / Ph | 0.1 | — | sparse | 5.596746 | 0.508340 | Run061 | no |
| 14M | T2 scalar / Ph | 0 | — | sparse | 5.127033 | 0.606922 | Run061 | no |
| 14M | T2 scalar / Ph | 0.1 | — | sparse | 5.152346 | 0.565242 | Run061 | yes |
| 14M | T7 tile / Ph | 0 | — | sparse | 5.721071 | 0.564432 | Run062 | no |
| 14M | T7 tile / Ph | 0.1 | — | sparse | 5.835102 | 0.516416 | Run062 | no |
| 14M | T7 scalar / Ph | 0 | — | sparse | 5.219827 | 0.635143 | Run062 | no |
| 14M | T7 scalar / Ph | 0.1 | — | sparse | 5.237395 | 0.556401 | Run062 | no |
| 14M | T2 tile / Ph | 0 | — | tile_generic_no_skip | 5.553770 | 0.646385 | Run063 | no |
| 14M | T2 tile / Ph | 0 | — | tile_generic_sparse | 5.553770 | 0.578467 | Run063 | no |
| 14M | T2 tile / Ph | 0 | — | tile_native | 5.553770 | 0.871060 | Run063 | no |
| 14M | T2 tile / Ph | 0 | — | tile_tile_no_skip | 5.553770 | 0.668422 | Run063 | no |
| 14M | T2 tile / Ph | 0 | — | tile_tile_sparse | 5.553770 | 0.517577 | Run063 | no |
| 14M | T2 tile / Ph | 0.1 | — | tile_generic_no_skip | 5.596746 | 0.643798 | Run063 | no |
| 14M | T2 tile / Ph | 0.1 | — | tile_generic_sparse | 5.596746 | 0.545248 | Run063 | no |
| 14M | T2 tile / Ph | 0.1 | — | tile_native | 5.596746 | 0.867335 | Run063 | no |
| 14M | T2 tile / Ph | 0.1 | — | tile_tile_no_skip | 5.596746 | 0.674537 | Run063 | no |
| 14M | T2 tile / Ph | 0.1 | — | tile_tile_sparse | 5.596746 | 0.496740 | Run063 | no |
| 14M | T2 tile / Ph | 0 | — | native | 5.553770 | 0.837750 | Run064 | no |
| 14M | T2 tile / Ph | 0 | — | generic_sparse | 5.553770 | 0.568726 | Run064 | no |
| 14M | T2 tile / Ph | 0 | — | generic_no_skip | 5.553770 | 0.635799 | Run064 | no |
| 14M | T2 tile / Ph | 0 | — | tile_sparse | 5.553770 | 0.509939 | Run064 | no |
| 14M | T2 tile / Ph | 0 | — | tile_no_skip | 5.553770 | 0.659606 | Run064 | no |
| 14M | T2 tile / Ph | 0.1 | — | native | 5.596746 | 0.834927 | Run064 | no |
| 14M | T2 tile / Ph | 0.1 | — | generic_sparse | 5.596746 | 0.534562 | Run064 | no |
| 14M | T2 tile / Ph | 0.1 | — | generic_no_skip | 5.596746 | 0.633721 | Run064 | no |
| 14M | T2 tile / Ph | 0.1 | — | tile_sparse | 5.596746 | 0.486441 | Run064 | yes |
| 14M | T2 tile / Ph | 0.1 | — | tile_no_skip | 5.596746 | 0.657777 | Run064 | no |

## FP16 native-body/head cohort

| Model | Training recipe | Body gate κ or pressure λ | Head κ | Execution | Loss | Full host ms | Session | 14M frontier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | T2 + corner 8x16 / Ph | 0 | 0 | custom_no_skip | 5.529001 | 1.127280 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph | 0 | 0 | custom_sparse | 5.529001 | 1.074223 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph | 0 | 0 | dense | 5.529001 | 0.998697 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph+head | 0 | 0 | custom_no_skip | 5.497717 | 1.127566 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph+head | 0 | 0 | custom_sparse | 5.497717 | 1.071257 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph+head | 0 | 0 | dense | 5.497717 | 0.999435 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph | 0.1 | 0.1 | custom_no_skip | 5.547977 | 1.127475 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph | 0.1 | 0.1 | custom_sparse | 5.547977 | 1.087872 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph | 0.1 | 0.1 | dense | 5.547977 | 0.999336 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph+head | 0.1 | 0.1 | custom_no_skip | 5.507129 | 1.129308 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph+head | 0.1 | 0.1 | custom_sparse | 5.507129 | 1.086614 | Run067 | no |
| 14M | T2 + corner 8x16 / Ph+head | 0.1 | 0.1 | dense | 5.507129 | 1.001191 | Run067 | no |
| 14M | T2 + corner 16x16 / Ph+head | 0 | 0.1 | custom_no_skip | 5.560725 | 1.109741 | Run068 | no |
| 14M | T2 + corner 16x16 / Ph+head | 0 | 0.1 | custom_sparse | 5.560725 | 0.994169 | Run068 | no |
| 14M | T2 + corner 16x16 / Ph+head | 0 | 0.1 | dense | 5.560725 | 0.983987 | Run068 | no |
| 14M | T2 + corner 16x16 / Ph+head | 0 | 0.5 | custom_no_skip | 5.882600 | 1.148920 | Run068 | no |
| 14M | T2 + corner 16x16 / Ph+head | 0 | 0.5 | custom_sparse | 5.882600 | 0.990146 | Run068 | no |
| 14M | T2 + corner 16x16 / Ph+head | 0 | 0.5 | dense | 5.882600 | 1.021812 | Run068 | no |

[Complete machine-readable coordinates](all-executions.csv) · [Source identities](../provenance/latency-sources.json).
