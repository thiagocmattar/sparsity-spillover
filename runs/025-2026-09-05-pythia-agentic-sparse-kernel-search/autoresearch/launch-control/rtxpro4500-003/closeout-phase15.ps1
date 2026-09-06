$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Lease = Get-Content -Raw (Join-Path $PSScriptRoot 'lease.json') | ConvertFrom-Json
$run025Record = Join-Path $PSScriptRoot 'closeout.json'
if (Test-Path -LiteralPath $run025Record) {
    throw 'Closeout record already exists; do not repeat a destructive teardown'
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
if (-not (Test-Path -LiteralPath $run025KnownHosts)) {
    throw 'Attempt-scoped SSH host key is missing'
}
$run025SshArgs = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
    '-o', 'StrictHostKeyChecking=yes', '-o', "UserKnownHostsFile=$run025KnownHosts",
    '-i', $run025Ssh.ssh_key.path, '-p', [string]$run025Ssh.port,
    ('root@' + $run025Ssh.ip)
)
$run025Remote = @'
set -euo pipefail
BASE=/workspace/run025-autoresearch-rtxpro4500-001
RUN="$BASE/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
PHASE="$RUN/artifacts/phase15-fresh-process-and-components-rtxpro4500-003"
test -f "$PHASE/all-checks-finished-utc.txt"
test "$(cat "$PHASE/exit-code.txt")" = 0
test ! -e "$BASE/run025-rtxpro4500-003-evidence.tar.gz"
"$RUN/artifacts/runtime/venv-pythia/bin/python" \
  "$RUN/autoresearch/launch-control/rtxpro4500-001/collect_evidence.py" \
  --build "$BASE/run025-rtxpro4500-003-evidence.tar.gz"
sha256sum "$BASE/run025-rtxpro4500-003-evidence.tar.gz"
'@
$run025RemoteOutput = & ssh.exe @run025SshArgs $run025Remote
if ($LASTEXITCODE -ne 0) { throw 'Remote completion check or evidence packaging failed' }
$run025RemoteOutput

$run025Destination = Join-Path $run025Run 'retrieved/rtxpro4500-003'
if (Test-Path -LiteralPath $run025Destination) {
    throw 'Retrieval destination already exists; refuse to merge evidence identities'
}
New-Item -ItemType Directory -Path $run025Destination | Out-Null
$run025Archive = Join-Path $run025Destination 'run025-rtxpro4500-003-evidence.tar.gz'
& scp.exe -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$run025KnownHosts" `
    -i $run025Ssh.ssh_key.path -P $run025Ssh.port `
    ("root@$($run025Ssh.ip):/workspace/run025-autoresearch-rtxpro4500-001/run025-rtxpro4500-003-evidence.tar.gz") `
    $run025Archive
if ($LASTEXITCODE -ne 0) { throw 'Evidence retrieval failed; Pod remains running' }
$run025Extracted = Join-Path $run025Destination 'evidence'
$run025Verifier = Join-Path $run025Run 'autoresearch/launch-control/rtxpro4500-001/collect_evidence.py'
$run025VerifyOutput = & python $run025Verifier --extract-verify $run025Archive --destination $run025Extracted
if ($LASTEXITCODE -ne 0) { throw 'Local evidence verification failed; Pod remains running' }
$run025Verify = $run025VerifyOutput | ConvertFrom-Json
$run025Hash = (Get-FileHash -LiteralPath $run025Archive -Algorithm SHA256).Hash.ToLowerInvariant()
$run025Bytes = (Get-Item -LiteralPath $run025Archive).Length

$run025DeleteOutput = & $run025Cli pod delete $run025Lease.pod_id 2>&1
if ($LASTEXITCODE -ne 0) {
    throw 'Evidence is verified, but Pod deletion was not confirmed'
}
$null = & $run025Cli pod get $run025Lease.pod_id 2>$null
if ($LASTEXITCODE -eq 0) {
    throw 'Pod still resolves after delete; inspect the control plane immediately'
}
$run025Closeout = [ordered]@{
    completed_utc = [DateTimeOffset]::UtcNow.ToString('o')
    pod_id = $run025Lease.pod_id
    pod_name = $run025Lease.name
    archive_sha256 = $run025Hash
    archive_bytes = $run025Bytes
    verified_files = [int]$run025Verify.verified_files
    remote_packaging_output = @($run025RemoteOutput)
    deletion_command_output = @($run025DeleteOutput)
    pod_get_after_delete = 'not found'
}
$run025Closeout | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $run025Record -Encoding UTF8
$run025Closeout | ConvertTo-Json -Depth 5
