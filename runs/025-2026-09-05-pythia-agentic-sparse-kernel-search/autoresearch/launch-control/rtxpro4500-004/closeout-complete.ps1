$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Lease = Get-Content -Raw (Join-Path $PSScriptRoot 'lease.json') | ConvertFrom-Json
$run025Record = Join-Path $PSScriptRoot 'closeout-complete.json'
if (Test-Path -LiteralPath $run025Record) {
    throw 'Complete closeout is already recorded; do not repeat destructive teardown'
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
$run025Remote = @'
set -euo pipefail
BASE=/workspace/run025-autoresearch-rtxpro4500-001
RUN="$BASE/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
PHASE17="$RUN/artifacts/phase17-fixed-rmodel-pairs-rtxpro4500-004"
PHASE18="$RUN/artifacts/phase18-14m-k001-confirmation-rtxpro4500-004"
ARCHIVE="$BASE/run025-rtxpro4500-004-complete-evidence.tar.gz"
test -f "$PHASE17/all-checks-finished-utc.txt"
test "$(cat "$PHASE17/exit-code.txt")" = 0
test -f "$PHASE18/all-checks-finished-utc.txt"
test "$(cat "$PHASE18/exit-code.txt")" = 0
test ! -e "$ARCHIVE"
mapfile -d '' ITEMS < <(
  find "$RUN/artifacts" -mindepth 1 -maxdepth 1 -type d \
    \( -name 'fixed-*-rtxpro4500-004' \
       -o -name 'k001fixed-*-rtxpro4500-004' \
       -o -name 'phase17-fixed-rmodel-pairs-rtxpro4500-004' \
       -o -name 'phase18-14m-k001-confirmation-rtxpro4500-004' \) \
    -printf 'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/artifacts/%f\0' | sort -z
)
if [[ "${#ITEMS[@]}" -ne 50 ]]; then
  printf 'Expected 50 Phase 17/18 artifact directories, found %s\n' "${#ITEMS[@]}" >&2
  exit 1
fi
tar -czf "$ARCHIVE" -C "$BASE/sparsity-spillover" "${ITEMS[@]}"
sha256sum "$ARCHIVE"
'@
$run025RemoteOutput = & ssh.exe @run025SshArgs $run025Remote
if ($LASTEXITCODE -ne 0) { throw 'Remote completion check or evidence packaging failed' }
$run025RemoteMatch = [regex]::Match(($run025RemoteOutput -join "`n"), '(?m)^([0-9a-f]{64})\s+')
if (-not $run025RemoteMatch.Success) { throw 'Remote archive SHA-256 was not reported' }

$run025Destination = Join-Path $run025Run 'retrieved/rtxpro4500-004'
if (Test-Path -LiteralPath $run025Destination) {
    throw 'Retrieval destination already exists; refuse to merge evidence identities'
}
New-Item -ItemType Directory -Path $run025Destination | Out-Null
$run025Archive = Join-Path $run025Destination 'run025-rtxpro4500-004-complete-evidence.tar.gz'
& scp.exe -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$run025KnownHosts" `
    -i $run025Ssh.ssh_key.path -P $run025Ssh.port `
    ("root@$($run025Ssh.ip):/workspace/run025-autoresearch-rtxpro4500-001/run025-rtxpro4500-004-complete-evidence.tar.gz") `
    $run025Archive
if ($LASTEXITCODE -ne 0) { throw 'Evidence retrieval failed; Pod remains running' }
$run025Hash = (Get-FileHash -LiteralPath $run025Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($run025Hash -ne $run025RemoteMatch.Groups[1].Value) {
    throw 'Local archive hash does not match the remote hash; Pod remains running'
}
$run025Phase17Verifier = Join-Path $PSScriptRoot 'verify_phase17_archive.py'
$run025Phase18Verifier = Join-Path $PSScriptRoot 'verify_phase18_archive.py'
$run025Phase17Text = & python $run025Phase17Verifier $run025Archive
if ($LASTEXITCODE -ne 0) { throw 'Phase 17 evidence verification failed; Pod remains running' }
$run025Phase18Text = & python $run025Phase18Verifier $run025Archive
if ($LASTEXITCODE -ne 0) { throw 'Phase 18 evidence verification failed; Pod remains running' }
$run025Phase17 = $run025Phase17Text | ConvertFrom-Json
$run025Phase18 = $run025Phase18Text | ConvertFrom-Json
[IO.File]::WriteAllText((Join-Path $run025Destination 'verification.json'), ($run025Phase17Text -join "`n") + "`n", [Text.Encoding]::UTF8)
[IO.File]::WriteAllText((Join-Path $run025Destination 'phase18-verification.json'), ($run025Phase18Text -join "`n") + "`n", [Text.Encoding]::UTF8)
$run025Bytes = (Get-Item -LiteralPath $run025Archive).Length

$run025DeleteOutput = & $run025Cli pod delete $run025Lease.pod_id 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Evidence is verified, but Pod deletion was not confirmed' }
$null = & $run025Cli pod get $run025Lease.pod_id 2>$null
if ($LASTEXITCODE -eq 0) { throw 'Pod still resolves after delete; inspect the control plane immediately' }
$run025Closeout = [ordered]@{
    completed_utc = [DateTimeOffset]::UtcNow.ToString('o')
    pod_id = $run025Lease.pod_id
    pod_name = $run025Lease.name
    archive_sha256 = $run025Hash
    archive_bytes = $run025Bytes
    phase17_processes = [int]$run025Phase17.processes
    phase17_candidate_validation_passes = [int]$run025Phase17.candidate_validation_passes
    phase18_processes = [int]$run025Phase18.processes
    phase18_candidate_validation_passes = [int]$run025Phase18.candidate_validation_passes
    fixed_rmodel_conditions = [int]$run025Phase17.fixed_rmodel_conditions + [int]$run025Phase18.fixed_rmodel_conditions
    deletion_command_output = @($run025DeleteOutput)
    pod_get_after_delete = 'not found'
}
$run025Closeout | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $run025Record -Encoding UTF8
$run025Closeout | ConvertTo-Json -Depth 5
