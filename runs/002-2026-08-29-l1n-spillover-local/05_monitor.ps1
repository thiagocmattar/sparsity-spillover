param(
    [Parameter(Mandatory = $true)]
    [int]$IntervalSeconds,
    [Parameter(Mandatory = $true)]
    [int]$CohortProcessId
)

$ErrorActionPreference = 'Stop'
$runDirectory = $PSScriptRoot
$artifactDirectory = Join-Path $runDirectory 'artifacts'
$progressPath = Join-Path $artifactDirectory 'progress.json'
$monitorPath = Join-Path $artifactDirectory 'monitoring.jsonl'

for ($milestone = 20; $milestone -le 100; $milestone += 20) {
    Start-Sleep -Seconds $IntervalSeconds
    $progress = $null
    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
    }
    $latestEvent = $null
    $attemptRoot = Join-Path $artifactDirectory 'attempts'
    if (Test-Path -LiteralPath $attemptRoot) {
        $latestAttempt = Get-ChildItem -LiteralPath $attemptRoot -Directory |
            Sort-Object Name |
            Select-Object -Last 1
        if ($null -ne $latestAttempt) {
            $eventPath = Join-Path $latestAttempt.FullName 'events.jsonl'
            if (Test-Path -LiteralPath $eventPath) {
                $line = Get-Content -LiteralPath $eventPath -Tail 1
                if ($line) {
                    $latestEvent = $line | ConvertFrom-Json
                }
            }
        }
    }
    $gpu = & nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    $process = Get-Process -Id $CohortProcessId -ErrorAction SilentlyContinue
    $snapshot = [ordered]@{
        timestamp = [DateTimeOffset]::UtcNow.ToString('o')
        planned_milestone_percent = $milestone
        interval_seconds = $IntervalSeconds
        process_running = $null -ne $process
        progress = $progress
        latest_event = $latestEvent
        gpu = $gpu
    }
    Add-Content -LiteralPath $monitorPath -Value ($snapshot | ConvertTo-Json -Depth 8 -Compress) -Encoding UTF8
    if ($null -ne $progress -and $progress.status -eq 'completed') {
        break
    }
}
