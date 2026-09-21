# Implementation details retained outside Appendix E

The 21 September 2026 appendix review moved these parameters out of the paper.
They describe the existing implementations; no kernel or measurement changed.

- The original 14M scalar path requires finite weights and biases. Nonzero
  magnitudes must lie in `[2^-50, 2^50]`; selected activations obey the same
  bounds. Rows failing the guards use the dense matrix path. The implementation
  accepts at most two nonzeros per scalar row and pads eight real rows to
  sixteen for matrix instructions. Sources: Run028
  [candidate.py](../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/candidate.py)
  and [joint.cu](../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/joint.cu).
- The initial 70M implementation changes projection dimensions from
  `(d, d_ff) = (128, 512)` to `(512, 2048)`. Its attention schedule uses
  `128 x 128` query/key tiles without KV splitting to match the reference
  reduction order; the 14M schedule uses `64 x 256` tiles and two partitions.
  Sources: [Run035 implementation](../runs/035-2026-09-18-pythia70m-k050-port/kernel/attention/kernel.cu)
  and [implementation audit](../analyses/024-2026-09-17-h-only-kernel-latency/observations/041-kernel-appendix.md).
- The optimized 70M implementation permits guarded scalar rows with at most
  eight nonzeros, accumulated in ascending feature order in FP32. Other rows
  retain the eight-row matrix path, with 256-column output tiles. Both
  inspection and projection launches remain inside timing. Sources:
  [Run042](../runs/042-2026-09-20-pythia70m-sparse-scale/README.md),
  [opt063](../runs/042-2026-09-20-pythia70m-sparse-scale/candidates/opt063/candidate.py)
  and its final [opt073 wrapper](../runs/042-2026-09-20-pythia70m-sparse-scale/candidates/opt073/candidate.py).

The removed sparsity/bypass plot and development history remain in their
original run/analysis records. The manuscript copy of the plot is retained at
`draft/.archive/figures/20-kernel-structure-native-base-speedup.pdf`.
The pre-review appendix is also preserved in commit `c593fd8e`.
