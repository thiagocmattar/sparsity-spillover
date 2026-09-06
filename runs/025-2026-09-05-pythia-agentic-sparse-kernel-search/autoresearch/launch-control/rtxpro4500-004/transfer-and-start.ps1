$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Bundle = Join-Path $run025Run 'autoresearch/bundles/phase17-code-001'
$run025Manifest = Get-Content -Raw (Join-Path $run025Bundle 'archive.json') | ConvertFrom-Json
$run025Lease = Get-Content -Raw (Join-Path $PSScriptRoot 'lease.json') | ConvertFrom-Json
$run025StartRecord = Join-Path $PSScriptRoot 'start.json'
if (Test-Path -LiteralPath $run025StartRecord) {
    throw 'Start record already exists; inspect the scoped worker instead of relaunching'
}
$run025Target = (& $run025Cli pod get $run025Lease.pod_id 2>$null) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $run025Target.name -ne $run025Lease.name) {
    throw 'Scoped RTX Pod is unavailable or has the wrong name'
}
$run025Ssh = (& $run025Cli ssh info $run025Lease.pod_id 2>$null) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $run025Ssh.ip -or -not $run025Ssh.port) {
    throw 'Scoped RTX Pod is not SSH-ready'
}
$run025KnownHosts = Join-Path $PSScriptRoot 'known_hosts'
if (Test-Path -LiteralPath $run025KnownHosts) {
    throw 'Attempt-scoped known_hosts already exists; refuse to overwrite it'
}
& ssh-keyscan.exe -p $run025Ssh.port $run025Ssh.ip 2>$null | Set-Content -LiteralPath $run025KnownHosts -Encoding ascii
if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $run025KnownHosts).Length -eq 0) {
    throw 'Unable to record the attempt-scoped SSH host key'
}
$run025SshArgs = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
    '-o', 'StrictHostKeyChecking=yes', '-o', "UserKnownHostsFile=$run025KnownHosts",
    '-i', $run025Ssh.ssh_key.path, '-p', [string]$run025Ssh.port,
    ('root@' + $run025Ssh.ip)
)
$run025RemotePrepare = @'
set -euo pipefail
test -d /workspace/run025-autoresearch-rtxpro4500-001/sparsity-spillover
test ! -e /workspace/run025-phase17-rtxpro4500-004
mkdir /workspace/run025-phase17-rtxpro4500-004
'@
& ssh.exe @run025SshArgs $run025RemotePrepare
if ($LASTEXITCODE -ne 0) { throw 'Remote persistent-volume preflight failed' }

$run025Archive = [string]$run025Manifest.path
if ((Get-Item -LiteralPath $run025Archive).Length -ne [long]$run025Manifest.expected_bytes) {
    throw 'Phase 17 bundle byte-count mismatch before transfer'
}
if ((Get-FileHash -LiteralPath $run025Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $run025Manifest.expected_sha256) {
    throw 'Phase 17 bundle hash mismatch before transfer'
}
$run025Sums = Join-Path $run025Bundle 'SHA256SUMS'
$run025Bootstrap = Join-Path $PSScriptRoot 'bootstrap-and-run.sh'
$run025TransferStarted = [DateTimeOffset]::UtcNow
& scp.exe -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$run025KnownHosts" `
    -i $run025Ssh.ssh_key.path -P $run025Ssh.port `
    $run025Archive $run025Sums $run025Bootstrap `
    ("root@$($run025Ssh.ip):/workspace/run025-phase17-rtxpro4500-004/")
if ($LASTEXITCODE -ne 0) { throw 'Small direct-SCP source transfer failed' }
$run025TransferFinished = [DateTimeOffset]::UtcNow

$run025RemoteStart = @'
set -euo pipefail
cd /workspace/run025-phase17-rtxpro4500-004
sha256sum -c SHA256SUMS
chmod 0755 bootstrap-and-run.sh
mkdir bootstrap-control
nohup setsid bash bootstrap-and-run.sh > bootstrap-control/outer.log 2>&1 < /dev/null &
printf '%s\n' "$!"
'@
$run025Pid = (& ssh.exe @run025SshArgs $run025RemoteStart | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or $run025Pid -notmatch '^\d+$') {
    throw 'Remote bootstrap did not return a process id'
}
$run025Start = [ordered]@{
    started_utc = [DateTimeOffset]::UtcNow.ToString('o')
    pod_id = $run025Lease.pod_id
    source_git_commit = $run025Manifest.git_commit
    transferred_bytes = [long]$run025Manifest.expected_bytes
    transferred_sha256 = $run025Manifest.expected_sha256
    transfer_started_utc = $run025TransferStarted.ToString('o')
    transfer_finished_utc = $run025TransferFinished.ToString('o')
    transfer_elapsed_seconds = [Math]::Round(($run025TransferFinished - $run025TransferStarted).TotalSeconds, 1)
    outer_pid = [int]$run025Pid
    route = 'direct SCP over attempt-scoped pinned SSH host key; no model or token-cache transfer'
}
$run025Start | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $run025StartRecord -Encoding UTF8
$run025Start | ConvertTo-Json -Depth 5
