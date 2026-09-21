# Environments

CPU review: Python 3.12, install the project with its dev extra. Matplotlib,
NumPy and pytest reconstruct measurements and check methods. The core runtime
pins are Torch 2.11.0, Transformers 5.12.1, NumPy 2.5.0, safetensors 0.8.0.

GPU measurement: Linux, NVIDIA RTX 5090, CUDA toolkit/runtime 12.8, a compatible
C++ compiler, Ninja and Python 3.12. Install Torch from its cu128 index before
installing this project:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]" ninja==1.11.1.4
python kernels/fetch_dependencies.py
```

`kernel-pip-freeze.txt` is the complete observed Linux kernel environment, not
a Windows lockfile. The historical container was
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`;
its base packages were updated to the recorded runtime before experiments.
Using that image alone does not install the paper environment.

The fetcher downloads only hash-checked immutable FlashAttention/CUTLASS source
archives and extracts headers and licenses. It does not fetch weights, install
credentials or create cloud resources. CUDA kernels compile on first use;
compilation and graph capture are excluded from timing.

Full training has substantial activation memory requirements. Historical 14M
MB32/OL1 workers reserved roughly 61 GB on 80 GB GPUs; do a representative
preflight on the selected device. Changing batch decomposition changes numerical
behavior and the recorded schedule identity even if effective batch is matched.
The release itself performs no GPU launch and makes no current price estimate.
