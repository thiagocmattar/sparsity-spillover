# Local GPU development environment

The user approved local setup on 23 September 2026. This is infrastructure
preparation, not a kernel experiment launch or a latency result. Keep the
Windows `.venv` and all prior run definitions unchanged.

## Host and current state

- RTX 5070 Ti Laptop GPU, 12,227 MiB total VRAM, approximately 9.5 GiB free at
  inspection; Windows driver 572.84. The prior 70M reference processes peaked
  at 5.26--5.54 GiB of PyTorch allocation on RTX 5090. This suggests local fit;
  reserved memory, compiler overhead and full-model fit still require a smoke.
- Windows build 26200.9457, approximately 64 GiB system RAM and 759 GiB free
  disk at inspection. Windows reports a hypervisor present.
- WSL 2.7.14.0 is installed from Microsoft's signed x64 MSI. Its SHA-256
  matches the official release asset; Authenticode reports Microsoft/Valid.
- VirtualMachinePlatform and Microsoft-Windows-Subsystem-Linux were enabled
  with `-NoRestart`. Both returned `RestartNeeded=True`.
- **A Windows restart is required before Linux/GPU setup can continue.** WSL
  status still reports virtualization unavailable in the current boot. Recheck
  after restarting; this does not yet establish a firmware problem.
- CUDA/PyTorch/Triton installation and GPU smoke have not run. Only shell syntax
  and Python compilation have been checked. No automatic restart is scheduled.

Installer receipts, the WSL installation log and the Ubuntu image are kept
outside Git under `%LOCALAPPDATA%\sparsity-spillover\gpu-setup`. The Windows
CPU-only environment remains usable. No GPU workload or RunPod restart was
performed for this setup.

## Resume after the user restarts Windows

Check `wsl --status` and `wsl --list --verbose`. Use the dedicated `SparsityGPU`
distribution, leaving any subsequently installed distributions alone. From
PowerShell, after verifying the Ubuntu image against `host-setup.json`:

```powershell
$taskRoot = Join-Path $env:LOCALAPPDATA 'sparsity-spillover'
$taskImage = Join-Path $taskRoot 'gpu-setup/ubuntu-24.04.5-wsl-amd64.wsl'
wsl.exe --import SparsityGPU (Join-Path $taskRoot 'wsl') $taskImage --version 2
$taskScript = (Resolve-Path tools/local_gpu/setup.sh).Path
$taskLinuxScript = (wsl.exe -d SparsityGPU -u root -- wslpath -u $taskScript).Trim()
wsl.exe -d SparsityGPU -u root -- bash $taskLinuxScript
```

Do not import again if `SparsityGPU` already exists. Investigate a partial
installation rather than deleting or unregistering it automatically. The setup
script requires the dedicated distro and Ubuntu 24.04. It installs the CUDA
12.8 toolkit without Linux GPU drivers, then recreates the pinned Run049 Python
3.12 / PyTorch 2.11.0+cu128 / Triton 3.6.0 / Transformers 5.12.1 environment.
The environment, compiler caches and setup logs live under `/opt/sparsity-gpu`
on the Linux filesystem. Subsequent commands source its `activate.sh`.

`smoke.py` checks CUDA visibility, BF16 matrix multiplication at the two 70M
projection shapes, Triton thresholding with changed-input CUDA graph replay,
and a compiled C++/CUDA extension. It reports versions and memory usage, but
does not qualify research kernels or establish full-model latency. A later
run must separately calibrate local resource fit and retain its own evidence.

Expected remaining setup time after reboot is approximately 15--30 minutes,
depending on package downloads and compilation. Local timings are a separate
hardware/environment cohort; final RTX 5090 claims need confirmation there.

## Sources

- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install).
- [WSL 2.7.14 release](https://github.com/microsoft/WSL/releases/tag/2.7.14).
- [Official distribution catalog](https://github.com/microsoft/WSL/blob/master/distributions/DistributionInfo.json).
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [PyTorch CUDA 12.8 packages](https://pytorch.org/get-started/previous-versions/).
