param(
    [int]$IntervalSeconds = 60,
    [int]$MaxPolls = 65
)

$ErrorActionPreference = 'Stop'
$runDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$progressPath = Join-Path $runDir 'artifacts\progress.json'
$artifacts = Join-Path $runDir 'artifacts'
$processIdPath = Join-Path $artifacts 'cohort.pid'
$stderrPath = Join-Path $artifacts 'cohort.stderr.log'

for ($poll = 1; $poll -le $MaxPolls; $poll++) {
    $snapshot = [ordered]@{ poll = $poll }
    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -Raw -LiteralPath $progressPath | ConvertFrom-Json
        $snapshot.progress = $progress
    } else {
        $progress = $null
        $snapshot.progress = @{ status = 'awaiting_progress' }
    }
    if (Test-Path -LiteralPath $processIdPath) {
        $cohortProcessId = [int](Get-Content -Raw -LiteralPath $processIdPath)
        $snapshot.process_id = $cohortProcessId
        $snapshot.process_alive = $null -ne (Get-Process -Id $cohortProcessId -ErrorAction SilentlyContinue)
    }
    $attemptsPath = Join-Path $artifacts 'attempts'
    if (Test-Path -LiteralPath $attemptsPath) {
        $latestAttempt = Get-ChildItem -LiteralPath $attemptsPath -Directory |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -ne $latestAttempt) {
            $manifestPath = Join-Path $latestAttempt.FullName 'manifest.json'
            $eventsPath = Join-Path $latestAttempt.FullName 'events.jsonl'
            if (Test-Path -LiteralPath $manifestPath) {
                $snapshot.manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
            }
            if (Test-Path -LiteralPath $eventsPath) {
                $latestEventLine = Get-Content -LiteralPath $eventsPath -Tail 1
                if ($latestEventLine) {
                    $snapshot.latest_event = $latestEventLine | ConvertFrom-Json
                    $snapshot.event_file_age_seconds = [math]::Round(
                        ((Get-Date).ToUniversalTime() - (Get-Item -LiteralPath $eventsPath).LastWriteTimeUtc).TotalSeconds,
                        1
                    )
                }
            }
        }
    }
    if (Test-Path -LiteralPath $stderrPath) {
        $snapshot.stderr_tail = @(Get-Content -LiteralPath $stderrPath -Tail 5)
    }
    $snapshot | ConvertTo-Json -Depth 12 -Compress
    if ($null -ne $progress -and $progress.status -in @('verified', 'failed')) {
        break
    }
    if ($poll -lt $MaxPolls) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}
