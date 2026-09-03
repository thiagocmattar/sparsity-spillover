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

$ErrorActionPreference = 'Continue'

function Write-GuardLog {
    param([string]$Message)
    $timestamp = [datetime]::UtcNow.ToString('o')
    Add-Content -LiteralPath $LogPath -Value "$timestamp $Message"
}

if (-not (Test-Path -LiteralPath $RunpodctlPath -PathType Leaf)) {
    throw "RunPod CLI does not exist at the exact requested path: $RunpodctlPath"
}

$deadline = $DeadlineUtc.ToUniversalTime()
Write-GuardLog "guard armed pod_id=$PodId deadline_utc=$($deadline.ToString('o')) cli=$RunpodctlPath"

while ([datetime]::UtcNow -lt $deadline) {
    $remaining = $deadline - [datetime]::UtcNow
    $sleepSeconds = [math]::Min(600, [math]::Max(1, [math]::Ceiling($remaining.TotalSeconds)))
    Start-Sleep -Seconds $sleepSeconds
}

$attempt = 0
while ($true) {
    $attempt += 1
    Write-GuardLog "deadline reached; delete attempt=$attempt pod_id=$PodId"
    $output = & $RunpodctlPath pod delete $PodId -o json 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    Write-GuardLog "delete attempt=$attempt exit_code=$exitCode output=$($output.Trim())"

    if ($exitCode -eq 0 -or $output -match '(?i)not found|does not exist') {
        Write-GuardLog "guard complete pod_id=$PodId"
        break
    }

    Write-GuardLog "delete failed; retrying in 60 seconds"
    Start-Sleep -Seconds 60
}
