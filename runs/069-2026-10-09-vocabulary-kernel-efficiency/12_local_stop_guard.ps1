param(
    [Parameter(Mandatory=$true)][string]$PodId,
    [Parameter(Mandatory=$true)][DateTimeOffset]$CreatedUtc,
    [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
    [Parameter(Mandatory=$true)][string]$RunpodctlPath,
    [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$') { throw 'Invalid Pod identity' }
if ($DeadlineUtc -le $CreatedUtc -or ($DeadlineUtc-$CreatedUtc).TotalHours -gt 4) {
    throw 'Deadline must be within four hours of Pod creation'
}
$run069Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
$run069ExpectedName = 'run069-vocabulary-efficiency-001'
function Read-Target {
    $run069Raw = & $run069Cli pod get $PodId 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
    $run069Target = ($run069Raw -join "`n") | ConvertFrom-Json
    if ($run069Target.name -ne $run069ExpectedName) { throw 'Pod name mismatch' }
    return $run069Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
    $run069Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
    Start-Sleep -Seconds $run069Seconds
}
for ($run069Try=1; $run069Try -le 5; $run069Try++) {
    try {
        $run069Before = Read-Target
        if ($run069Before.runtimeStatus -notin @('stopped','terminated') -and
            $run069Before.desiredStatus -notin @('EXITED','STOPPED','TERMINATED')) {
            $run069StopOutput = & $run069Cli pod stop $PodId 2>&1
            if ($LASTEXITCODE -ne 0) { throw 'Provider stop failed' }
        }
        $run069After = Read-Target
        if ($run069After.runtimeStatus -notin @('stopped','terminated') -and
            $run069After.desiredStatus -notin @('EXITED','STOPPED','TERMINATED')) {
            throw 'Provider stop unconfirmed'
        }
        Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; workspace retained for verified retrieval"
        exit 0
    } catch {
        Add-Content -LiteralPath $LogPath -Value "RETRY $run069Try provider stop $PodId"
        Start-Sleep -Seconds 15
    }
}
Add-Content -LiteralPath $LogPath -Value "STOP_UNCONFIRMED $PodId; operator intervention required"
exit 1
