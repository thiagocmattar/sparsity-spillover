param(
    [Parameter(Mandatory = $true)]
    [string]$PodId,

    [Parameter(Mandatory = $true)]
    [datetime]$DeadlineUtc,

    [Parameter(Mandatory = $true)]
    [string]$RunpodctlPath,

    [Parameter(Mandatory = $true)]
    [string]$LogPath
)

$ErrorActionPreference = 'Stop'

function Write-GuardLog {
    param([string]$Message)
    $timestamp = [datetime]::UtcNow.ToString('o')
    Add-Content -LiteralPath $LogPath -Value "$timestamp $Message"
}

Write-GuardLog "guard armed pod_id=$PodId deadline_utc=$($DeadlineUtc.ToUniversalTime().ToString('o'))"

while ([datetime]::UtcNow -lt $DeadlineUtc.ToUniversalTime()) {
    $remaining = $DeadlineUtc.ToUniversalTime() - [datetime]::UtcNow
    $sleepSeconds = [math]::Min(60, [math]::Max(1, [math]::Ceiling($remaining.TotalSeconds)))
    Start-Sleep -Seconds $sleepSeconds
}

Write-GuardLog "deadline reached; deleting pod_id=$PodId"
$output = & $RunpodctlPath pod delete $PodId 2>&1 | Out-String
$exitCode = $LASTEXITCODE
Write-GuardLog "delete exit_code=$exitCode output=$($output.Trim())"
