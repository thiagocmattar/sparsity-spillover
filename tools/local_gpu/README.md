# Local GPU development environment

The user approved local setup on 23 September 2026. This is infrastructure
preparation, not a kernel experiment launch or a latency result. Keep the
Windows `.venv` and all prior run definitions unchanged.

## Verified state: 23 September 2026

**Ready for local kernel development.** After the user's Windows restart, the
dedicated `SparsityGPU` WSL2 distribution was imported from the verified Ubuntu
24.04 image. The normal Linux user is `researcher`.

- RTX 5070 Ti Laptop GPU, 12,227 MiB VRAM, compute capability 12.0;
  Windows NVIDIA driver 572.84.
- Windows build 26200.9457, approximately 64 GiB system RAM; WSL 2.7.14.0.
- Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0,
  Transformers 5.12.1 and NumPy 2.5.0. All 58 package pins from Run049 match.
- BF16 matrix multiplication passed at `(M,K,N)=(2048,512,512)` and
  `(2048,2048,512)`, with relative L2 errors approximately 0.00166.
- Triton thresholding passed exact comparison on changed-input CUDA graph
  replay. A C++/CUDA extension compiled for `sm_120` and passed exact comparison.
  These checks ran as `researcher`, not root.
- Windows `.venv` remains PyTorch 2.11.0+cpu. Prior run definitions are unchanged.

[smoke.json](smoke.json) contains the GPU checks.
[wsl-bootstrap.json](wsl-bootstrap.json) records package/isolation verification,
source hashes and hashes of seven retrieved log files. Their copies were
verified byte-for-byte under
`%LOCALAPPDATA%\sparsity-spillover\gpu-setup\linux-logs`.
[host-setup.json](host-setup.json) is the historical **pre-restart** receipt;
its pending status is superseded by the WSL verification record.

This verifies the development environment, not a research kernel or full-model
fit. The prior 70M reference processes allocated 5.26--5.54 GiB on RTX 5090;
local full-model memory, numerical qualification and ETC still need calibration.
Local latency is a separate hardware/environment cohort. Final RTX 5090 claims
need confirmation there. No new experiment or RunPod restart was performed.

## Storage and Windows access

The Linux disk is under `%LOCALAPPDATA%\sparsity-spillover\wsl`. Environment,
compiler caches and installation logs are under `/opt/sparsity-gpu` inside Linux.
Installer files remain outside Git under
`%LOCALAPPDATA%\sparsity-spillover\gpu-setup`.

`/etc/wsl.conf` disables automatic Windows-drive mounting, fstab processing,
Windows executable interop and Windows PATH propagation. Verification found no
mounted `/mnt/<drive>` paths or interop endpoint. GPU driver support remains
available through WSL's driver mounts; this is not a strict security sandbox.
Only the explicit setup files and pinned requirements were copied through stdin;
the Windows repository was not mounted. Future source/data transfers must be
explicit copies with their own identity checks.

## Use the installed environment

Open the dedicated distro from PowerShell:

```powershell
wsl.exe -d SparsityGPU --cd /home/researcher
```

Then in Linux:

```bash
source /opt/sparsity-gpu/activate.sh
python -c 'import torch; print(torch.cuda.get_device_name())'
```

To repeat only the small infrastructure check:

```bash
python /opt/sparsity-gpu/setup/smoke.py \
  --output /home/researcher/.local/state/sparsity-gpu/smoke-repeat.json
```

## Setup implementation and corrections

`bootstrap_wsl.py` verifies the downloaded image, imports the distro only when
absent, configures access, transfers three setup files with LF line endings,
and runs `setup.sh`. It restarts this distro to apply configuration, so use it
only when the distro is idle. Do not rerun bootstrap just to use the environment.
It refuses to overwrite a nonempty unregistered distro directory.

The initial Linux `nohup` launcher did not keep WSL running after the Windows
caller exited. The corrected launcher keeps the Windows WSL client alive until
setup finishes. The first actual setup reached Triton, then failed because
`Python.h` was missing. Adding `python3.12-dev` fixed this; the second setup
completed with exit code zero. Both the original failure and final success are
retained in the installation log. Shell syntax and both Python sources also pass
syntax checks.

`setup.sh` installs `cuda-toolkit-12-8` without a Linux NVIDIA driver, installs
the pinned Python packages and runs the smoke before reporting readiness.
Keep its terminal/process alive through installation; it logs persistently to
`/opt/sparsity-gpu/logs/setup.log`. Research jobs need a separate persistent
launcher and run-owned logs.

## Sources

- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install).
- [Microsoft WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).
- [WSL 2.7.14 release](https://github.com/microsoft/WSL/releases/tag/2.7.14).
- [Official distribution catalog](https://github.com/microsoft/WSL/blob/master/distributions/DistributionInfo.json).
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [PyTorch CUDA 12.8 packages](https://pytorch.org/get-started/previous-versions/).
