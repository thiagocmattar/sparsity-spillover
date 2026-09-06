#!/usr/bin/env bash
set -euo pipefail
run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_control="$run025_base/runpodctl-install"
mkdir "$run025_control"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
curl -fL --max-time 300 -o "$run025_control/runpodctl-v2.12.0.download" https://github.com/runpod/runpodctl/releases/download/v2.12.0/runpodctl-linux-amd64
# Official v2.12.0 checksums_2.12.0_sha256.txt, Linux amd64 asset.
printf '%s  %s\n' f273555b935963925e696e95f36a883ca68c5c845efc893db9f8f701749c8474 "$run025_control/runpodctl-v2.12.0.download" | sha256sum -c -
install -m 0755 "$run025_control/runpodctl-v2.12.0.download" "$run025_base/runpodctl-v2.12.0"
"$run025_base/runpodctl-v2.12.0" version > "$run025_control/version.txt"
