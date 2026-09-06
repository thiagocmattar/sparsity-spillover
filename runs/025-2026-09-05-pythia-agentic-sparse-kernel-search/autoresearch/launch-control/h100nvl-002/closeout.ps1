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
    throw 'Scoped H100 Pod is unavailable or has the wrong name'
}
$run025Ssh = (& $run025Cli ssh info $run025Lease.pod_id 2>$null) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $run025Ssh.ip -or -not $run025Ssh.port) {
    throw 'Scoped H100 Pod is not SSH-ready'
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
BASE=/workspace/run025-h100nvl-001
test -f "$BASE/bootstrap-control/evidence-ready-utc.txt"
test -f "$BASE/bootstrap-control/finished-utc.txt"
test "$(cat "$BASE/bootstrap-control/exit-code.txt")" = 0
test -f "$BASE/run025-h100nvl-001-evidence.tar"
test -f "$BASE/run025-h100nvl-001-evidence.tar.sha256"
stat -c '%s' "$BASE/run025-h100nvl-001-evidence.tar"
sha256sum "$BASE/run025-h100nvl-001-evidence.tar"
'@
$run025RemoteOutput = @(& ssh.exe @run025SshArgs $run025Remote)
if ($LASTEXITCODE -ne 0 -or $run025RemoteOutput.Count -lt 2) {
    throw 'Remote H100 completion or evidence check failed'
}
$run025RemoteBytes = [long]$run025RemoteOutput[0]
$run025RemoteHash = ([string]$run025RemoteOutput[1]).Split(' ', [StringSplitOptions]::RemoveEmptyEntries)[0]
if ($run025RemoteHash -notmatch '^[0-9a-f]{64}$') { throw 'Invalid remote evidence hash' }

$run025Destination = Join-Path $run025Run 'retrieved/h100nvl-002'
if (Test-Path -LiteralPath $run025Destination) {
    throw 'Retrieval destination already exists; refuse to merge evidence identities'
}
New-Item -ItemType Directory -Path $run025Destination | Out-Null
$run025Archive = Join-Path $run025Destination 'run025-h100nvl-002-evidence.tar'
& scp.exe -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$run025KnownHosts" `
    -i $run025Ssh.ssh_key.path -P $run025Ssh.port `
    ("root@$($run025Ssh.ip):/workspace/run025-h100nvl-001/run025-h100nvl-001-evidence.tar") `
    $run025Archive
if ($LASTEXITCODE -ne 0) { throw 'H100 evidence retrieval failed; Pod remains running' }
& scp.exe -q -r -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$run025KnownHosts" `
    -i $run025Ssh.ssh_key.path -P $run025Ssh.port `
    ("root@$($run025Ssh.ip):/workspace/run025-h100nvl-001/bootstrap-control") `
    $run025Destination
if ($LASTEXITCODE -ne 0) { throw 'H100 bootstrap record retrieval failed; Pod remains running' }

$run025LocalBytes = (Get-Item -LiteralPath $run025Archive).Length
$run025LocalHash = (Get-FileHash -LiteralPath $run025Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($run025LocalBytes -ne $run025RemoteBytes -or $run025LocalHash -ne $run025RemoteHash) {
    throw 'Retrieved H100 archive does not match remote identity; Pod remains running'
}
$run025Extracted = Join-Path $run025Destination 'evidence-verified'
$run025Verifier = Join-Path $PSScriptRoot '../h100nvl-001/verify_h100_evidence.py'
$run025Extended = '\\?\' + [IO.Path]::GetFullPath($run025Extracted)
$run025VerifyOutput = & python $run025Verifier $run025Archive $run025Extended
if ($LASTEXITCODE -ne 0) { throw 'Local H100 evidence verification failed; Pod remains running' }
$run025Verify = $run025VerifyOutput | ConvertFrom-Json

$run025DeleteOutput = & $run025Cli pod delete $run025Lease.pod_id 2>&1
if ($LASTEXITCODE -ne 0) {
    throw 'H100 evidence is verified, but Pod deletion was not confirmed'
}
$null = & $run025Cli pod get $run025Lease.pod_id 2>$null
if ($LASTEXITCODE -eq 0) {
    throw 'H100 Pod still resolves after delete; inspect the control plane immediately'
}
$run025Closeout = [ordered]@{
    completed_utc = [DateTimeOffset]::UtcNow.ToString('o')
    pod_id = $run025Lease.pod_id
    pod_name = $run025Lease.name
    archive_sha256 = $run025LocalHash
    archive_bytes = $run025LocalBytes
    verified_files = [int]$run025Verify.verified_files
    artifact_directories = [int]$run025Verify.artifact_directories
    deletion_command_output = @($run025DeleteOutput)
    pod_get_after_delete = 'not found'
}
$run025Closeout | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $run025Record -Encoding UTF8
$run025Closeout | ConvertTo-Json -Depth 5
