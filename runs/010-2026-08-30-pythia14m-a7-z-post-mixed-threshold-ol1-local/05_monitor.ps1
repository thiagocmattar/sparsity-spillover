param(
    [int]$IntervalSeconds = 1800,
    [int]$MaxPolls = 4
)

$ErrorActionPreference = 'Stop'
$runDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$artifacts = Join-Path $runDir 'artifacts'
$progressPath = Join-Path $artifacts 'progress.json'
$driverPath = Join-Path $artifacts 'driver.json'
$processIdPath = Join-Path $artifacts 'cohort.pid'
$stderrPath = Join-Path $artifacts 'cohort.stderr.log'
$attemptsPath = Join-Path $artifacts 'attempts'
$totalSteps = 2905

function Read-SharedText([string]$Path) {
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite
    )
    try {
        $reader = [System.IO.StreamReader]::new($stream)
        try {
            return $reader.ReadToEnd()
        } finally {
            $reader.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

for ($poll = 1; $poll -le $MaxPolls; $poll++) {
    $snapshot = [ordered]@{ poll = $poll; total_optimizer_steps = $totalSteps }
    $progress = $null
    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -Raw -LiteralPath $progressPath | ConvertFrom-Json
        $snapshot.status = $progress.status
        $snapshot.current_condition = $progress.current_condition
    } else {
        $snapshot.status = 'awaiting_progress'
    }
    if (Test-Path -LiteralPath $processIdPath) {
        $cohortProcessId = [int](Get-Content -Raw -LiteralPath $processIdPath)
        $snapshot.process_id = $cohortProcessId
        $snapshot.process_alive = $null -ne (Get-Process -Id $cohortProcessId -ErrorAction SilentlyContinue)
    }

    $completedSteps = 0
    $latestTrain = $null
    $latestValidation = $null
    if (Test-Path -LiteralPath $attemptsPath) {
        $attemptDirs = @(Get-ChildItem -LiteralPath $attemptsPath -Directory | Sort-Object Name)
        foreach ($attemptDir in $attemptDirs) {
            $manifestPath = Join-Path $attemptDir.FullName 'manifest.json'
            $eventsPath = Join-Path $attemptDir.FullName 'events.jsonl'
            $manifest = $null
            if (Test-Path -LiteralPath $manifestPath) {
                $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
            }
            $trainForAttempt = $null
            if (Test-Path -LiteralPath $eventsPath) {
                $eventLines = @((Read-SharedText $eventsPath) -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
                $tailEvents = @($eventLines | Select-Object -Last 12 | ForEach-Object { $_ | ConvertFrom-Json })
                $trainForAttempt = $tailEvents | Where-Object { $_.event -eq 'train' } | Select-Object -Last 1
                $validationForAttempt = $tailEvents | Where-Object { $_.event -eq 'validation' } | Select-Object -Last 1
                if ($null -ne $trainForAttempt) { $latestTrain = $trainForAttempt }
                if ($null -ne $validationForAttempt) { $latestValidation = $validationForAttempt }
                $snapshot.event_file_age_seconds = [math]::Round(
                    ((Get-Date).ToUniversalTime() - (Get-Item -LiteralPath $eventsPath).LastWriteTimeUtc).TotalSeconds,
                    1
                )
            }
            if ($null -ne $manifest -and $manifest.status -eq 'completed') {
                $completedSteps += [int]$manifest.completed_steps
            } elseif ($null -ne $trainForAttempt) {
                $completedSteps += [int]$trainForAttempt.step
            }
        }
    }
    $snapshot.completed_optimizer_steps = $completedSteps
    $snapshot.progress_percent = [math]::Round(100.0 * $completedSteps / $totalSteps, 2)
    if ($null -ne $latestTrain) {
        $snapshot.latest_condition = $latestTrain.condition_id
        $snapshot.latest_step = $latestTrain.step
        $snapshot.task_loss = $latestTrain.task_loss
        $snapshot.throughput_tokens_per_second = $latestTrain.tokens_per_second
    }
    if ($null -ne $latestValidation) {
        $snapshot.latest_validation_loss = $latestValidation.loss
        $snapshot.latest_validation_source = $latestValidation.source
    }
    if (Test-Path -LiteralPath $driverPath) {
        $driver = Get-Content -Raw -LiteralPath $driverPath | ConvertFrom-Json
        $startedAt = [datetimeoffset]::Parse([string]$driver.started_at).ToUniversalTime()
        $elapsed = ([datetimeoffset]::UtcNow - $startedAt).TotalSeconds
        $snapshot.elapsed_seconds = [math]::Round($elapsed, 1)
        if ($completedSteps -gt 0 -and $completedSteps -lt $totalSteps) {
            $snapshot.etc_seconds = [math]::Round($elapsed * ($totalSteps / $completedSteps - 1.0), 1)
        } elseif ($completedSteps -ge $totalSteps) {
            $snapshot.etc_seconds = 0.0
        }
    }
    if (Test-Path -LiteralPath $stderrPath) {
        $stderrLines = @((Read-SharedText $stderrPath) -split "`r?`n")
        $snapshot.stderr_tail = @($stderrLines | Select-Object -Last 5)
    }
    $snapshot | ConvertTo-Json -Depth 12 -Compress
    if ($null -ne $progress -and $progress.status -in @('verified', 'failed')) {
        break
    }
    if ($poll -lt $MaxPolls) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}
