#!/usr/bin/env bash
set -eu
control=/workspace/run068-control
curl -fsSL --retry 3 https://github.com/runpod/runpodctl/releases/download/v2.14.0/runpodctl-linux-amd64 -o "$control/runpodctl-v2.14.0"
printf '%s  %s\n' 2e0fd370a52a0fc7e43a6434a209348a4f6836fcdf1ad2b093b609e938138be9 "$control/runpodctl-v2.14.0" | sha256sum -c -
chmod 700 "$control/runpodctl-v2.14.0"
"$control/runpodctl-v2.14.0" version
"$control/runpodctl-v2.14.0" receive --help
mkdir -p "$control/incoming-003"
