$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Lease = Get-Content -Raw (Join-Path $PSScriptRoot 'lease.json') | ConvertFrom-Json
$run025Record = Join-Path $PSScriptRoot 'phase18-start.json'
if (Test-Path -LiteralPath $run025Record) {
    throw 'Phase 18 start is already recorded'
}
$run025Remaining = [DateTimeOffset]$run025Lease.stop_deadline_utc - [DateTimeOffset]::UtcNow
if ($run025Remaining.TotalMinutes -lt 45) {
    throw 'Less than 45 minutes remain on the deletion guard; do not start Phase 18'
}
$run025Target = (& $run025Cli pod get $run025Lease.pod_id 2>$null) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $run025Target.name -ne $run025Lease.name) {
    throw 'Scoped RTX Pod is unavailable or has the wrong name'
}
$run025Ssh = (& $run025Cli ssh info $run025Lease.pod_id 2>$null) | ConvertFrom-Json
$run025KnownHosts = Join-Path $PSScriptRoot 'known_hosts'
if (-not $run025Ssh.ip -or -not $run025Ssh.port -or -not (Test-Path -LiteralPath $run025KnownHosts)) {
    throw 'Pinned SSH route is unavailable'
}
$run025SshArgs = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
    '-o', 'StrictHostKeyChecking=yes', '-o', "UserKnownHostsFile=$run025KnownHosts",
    '-i', $run025Ssh.ssh_key.path, '-p', [string]$run025Ssh.port,
    ('root@' + $run025Ssh.ip)
)
$run025Source = Join-Path $PSScriptRoot 'phase18-14m-k001-confirmation.sh'
$run025Hash = (Get-FileHash -LiteralPath $run025Source -Algorithm SHA256).Hash.ToLowerInvariant()
$run025Bytes = (Get-Item -LiteralPath $run025Source).Length
$run025RemotePrepare = @'
set -euo pipefail
RUN=/workspace/run025-autoresearch-rtxpro4500-001/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
PHASE17="$RUN/artifacts/phase17-fixed-rmodel-pairs-rtxpro4500-004"
test -f "$PHASE17/all-checks-finished-utc.txt"
test "$(cat "$PHASE17/exit-code.txt")" = 0
test ! -e /workspace/run025-phase18-rtxpro4500-004
mkdir /workspace/run025-phase18-rtxpro4500-004
'@
& ssh.exe @run025SshArgs $run025RemotePrepare
if ($LASTEXITCODE -ne 0) { throw 'Phase 17 is not complete or Phase 18 staging already exists' }
$run025TransferStarted = [DateTimeOffset]::UtcNow
& scp.exe -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$run025KnownHosts" `
    -i $run025Ssh.ssh_key.path -P $run025Ssh.port $run025Source `
    ("root@$($run025Ssh.ip):/workspace/run025-phase18-rtxpro4500-004/")
if ($LASTEXITCODE -ne 0) { throw 'Phase 18 controller transfer failed' }
$run025TransferFinished = [DateTimeOffset]::UtcNow
$run025RemoteStart = @"
set -euo pipefail
BASE=/workspace/run025-phase18-rtxpro4500-004
printf '%s  %s\n' '$run025Hash' "`$BASE/phase18-14m-k001-confirmation.sh" | sha256sum -c -
mkdir "`$BASE/control"
chmod 0755 "`$BASE/phase18-14m-k001-confirmation.sh"
nohup setsid bash "`$BASE/phase18-14m-k001-confirmation.sh" > "`$BASE/control/outer.log" 2>&1 < /dev/null &
printf '%s\n' "`$!"
"@
$run025Pid = (& ssh.exe @run025SshArgs $run025RemoteStart | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or $run025Pid -notmatch '^\d+$') {
    throw 'Phase 18 did not return a process id'
}
$run025Commit = (& git -C $run025Root rev-parse HEAD).Trim()
$run025Start = [ordered]@{
    started_utc = [DateTimeOffset]::UtcNow.ToString('o')
    pod_id = $run025Lease.pod_id
    source_git_commit = $run025Commit
    controller_bytes = $run025Bytes
    controller_sha256 = $run025Hash
    transfer_elapsed_seconds = [Math]::Round(($run025TransferFinished - $run025TransferStarted).TotalSeconds, 1)
    outer_pid = [int]$run025Pid
    parent_phase = 'phase17-fixed-rmodel-pairs-rtxpro4500-004'
    reason = 'direct fresh-process confirmation of the P0-to-K001 kernel optimization after K013 did not beat P0 on the first Phase 17 14M endpoint'
}
$run025Start | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $run025Record -Encoding UTF8
$run025Start | ConvertTo-Json -Depth 4
