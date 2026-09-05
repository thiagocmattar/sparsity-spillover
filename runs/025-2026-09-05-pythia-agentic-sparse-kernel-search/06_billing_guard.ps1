param(
    [Parameter(Mandatory=$true)][string]$PodId,
    [Parameter(Mandatory=$true)][string]$ExpectedName,
    [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
    [Parameter(Mandatory=$true)][string]$RunpodctlPath,
    [Parameter(Mandatory=$true)][string]$LogPath,
    [switch]$DryRun
)
# Local independent stop guard. Never deletes storage or injects a key into a pod.
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run025-*') {
    throw 'Invalid scoped Run 025 pod identity'
}
$run025Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
$run025Log = [IO.Path]::GetFullPath($LogPath)
function Write-GuardLog([string]$message) {
    Add-Content -LiteralPath $run025Log -Value "$([DateTimeOffset]::UtcNow.ToString('o')) $message"
}
if ($DryRun) {
    Write-GuardLog "DRY_RUN would stop only $PodId named $ExpectedName at $($DeadlineUtc.ToString('o')); no API call"
    exit 0
}
function Read-Target {
    $run025Raw = & $run025Cli pod get $PodId 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Scoped pod lookup failed; inspect control plane' }
    $run025Target = $run025Raw | ConvertFrom-Json
    if ($run025Target.name -ne $ExpectedName) { throw 'Pod name does not match the approved lease' }
    return $run025Target
}
$null = Read-Target
Write-GuardLog "ARMED stop guard for $PodId; requires local machine to remain awake"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
    $run025Wait = [Math]::Min(600, [Math]::Max(1, [Math]::Ceiling(($DeadlineUtc - [DateTimeOffset]::UtcNow).TotalSeconds)))
    Start-Sleep -Seconds $run025Wait
}
for ($run025Try = 1; $run025Try -le 3; $run025Try++) {
    try {
        $null = Read-Target
        $run025StopOutput = & $run025Cli pod stop $PodId 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'Pod stop failed' }
        $run025After = Read-Target
        if ($run025After.desiredStatus -notin @('EXITED', 'STOPPED')) { throw 'Stop not yet confirmed' }
        Write-GuardLog "STOP_CONFIRMED $PodId; retained disk still bills; retrieve, hash-verify, then delete"
        exit 0
    } catch {
        Write-GuardLog "STOP_RETRY $run025Try for $PodId; inspect control plane if retries exhaust"
        if ($run025Try -lt 3) { Start-Sleep -Seconds 15 }
    }
}
Write-GuardLog "STOP_UNCONFIRMED $PodId; urgent manual/control-plane check required"
exit 1
