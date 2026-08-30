param(
    [int]$IntervalSeconds = 300,
    [int]$StaleEventSeconds = 420
)

$runRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$progressPath = Join-Path $runRoot 'artifacts\progress.json'
$driverPath = Join-Path $runRoot 'artifacts\driver.json'
$attemptsRoot = Join-Path $runRoot 'artifacts\attempts'

while ($true) {
    Start-Sleep -Seconds $IntervalSeconds
    $timestamp = (Get-Date).ToUniversalTime().ToString('o')
    $latestStep = $null
    $latestTaskLoss = $null
    $latestPressureLoss = $null
    $latestGradientNorm = $null
    $latestTokensPerSecond = $null
    $eventAgeSeconds = $null
    $warnings = @()
    if (Test-Path -LiteralPath $attemptsRoot) {
        $latestEvents = Get-ChildItem -LiteralPath $attemptsRoot -Recurse -Filter 'events.jsonl' -File |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -ne $latestEvents) {
            $eventAgeSeconds = ((Get-Date).ToUniversalTime() - $latestEvents.LastWriteTimeUtc).TotalSeconds
            $latestLine = Get-Content -LiteralPath $latestEvents.FullName -Tail 1
            if ($latestLine) {
                $latestEvent = $latestLine | ConvertFrom-Json
                $latestStep = $latestEvent.step
                $latestTaskLoss = $latestEvent.task_loss
                $latestPressureLoss = $latestEvent.pressure_loss
                $latestGradientNorm = $latestEvent.adamw_gradient_norm_pre_clip
                $latestTokensPerSecond = $latestEvent.tokens_per_second
            }
            if ($eventAgeSeconds -gt $StaleEventSeconds) {
                $warnings += "No durable event for $([Math]::Round($eventAgeSeconds)) seconds"
            }
        }
    }
    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
        [pscustomobject]@{
            checked_at = $timestamp
            status = $progress.status
            completed_conditions = $progress.completed_conditions
            condition_count = $progress.condition_count
            current_condition = $progress.current_condition
            elapsed_seconds = $progress.elapsed_seconds
            latest_step = $latestStep
            latest_task_loss = $latestTaskLoss
            latest_pressure_loss = $latestPressureLoss
            latest_combined_gradient_norm = $latestGradientNorm
            latest_tokens_per_second = $latestTokensPerSecond
            latest_event_age_seconds = $eventAgeSeconds
            warnings = $warnings
        } | ConvertTo-Json -Compress
        if ($progress.status -in @('verified', 'failed')) {
            break
        }
    }
    if (Test-Path -LiteralPath $driverPath) {
        $driver = Get-Content -LiteralPath $driverPath -Raw | ConvertFrom-Json
        if ($driver.status -in @('completed', 'failed')) {
            break
        }
    }
}
