# Final specialized execution

`install.py` directly assembles one fixed implementation for each model size.
There is no candidate registry or chain of older installers.

| Component | 14M | 70M |
|---|---|---|
| Normalization | Fused pair, separate affine parameters and existing a/m gates | Same policy, width 512 |
| QKV and FFN-up | CUTLASS; skip empty activation fragments | Native PyTorch linear |
| h,z output projections | Eight-row inspection, short-row scalar path, sparse MMA fallback | Parallel inspection, selected eight-row scalar policy and N256 output tiles |
| Attention | Exact-zero QK/PV MMA bypass; two KV partitions | Dense final attention schedule, token-major output |
| RoPE and QKV gates | Shared `rotary/` implementation | Shared `rotary/` implementation |
| Vocabulary projection | Native dense projection | Fixed CUTLASS 128×128×32 tile, all 50,304 logits |

`model_14m/` and `model_70m/` contain the selected Python wrappers, CUDA files and
required attention headers. `ablation/` contains the independent six-path
execution switches and counters used for Table 2. These are measured paper
controls, not optimization candidates. `controls.py` supplies the native h,z
replacement while retaining the same checkpoint and gates.

The implementation expects Linux, RTX 5090, BF16 inference, batch one, sequence
length 2,048, fixed weights and non-overlapping buffer use. It does not support
training, cached decoding or concurrent serving. Headers are fetched with
`python kernels/fetch_dependencies.py`; see `environment/README.md`.

The selected arithmetic and launch parameters are retained from the measured
implementations. Assembly and names were simplified for this release. This
packaging change has CPU checks but still requires full GPU qualification before
claiming new timing results; `scripts/benchmark.py` enforces that numerical gate.
