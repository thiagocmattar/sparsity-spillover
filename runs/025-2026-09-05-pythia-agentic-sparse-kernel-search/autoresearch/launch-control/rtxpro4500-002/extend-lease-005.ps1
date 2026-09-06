$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Lease = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'lease.json') -Raw | ConvertFrom-Json
$run025Prior = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'lease-extension-004.json') -Raw | ConvertFrom-Json
$run025RecordPath = Join-Path $PSScriptRoot 'lease-extension-005.json'
$run025Log = Join-Path $PSScriptRoot 'billing-guard-extension-005.log'
$run025ErrorLog = Join-Path $PSScriptRoot 'guard-extension-005-error.log'
if (Test-Path -LiteralPath $run025RecordPath) { throw 'Lease extension already recorded' }
$run025OldGuard = Get-Process -Id ([int]$run025Prior.replacement_guard_pid) -ErrorAction Stop
$run025Target = (& $run025Cli pod get $run025Lease.pod_id 2>$null) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $run025Target.name -ne $run025Lease.name) {
    throw 'Scoped Pod identity mismatch before extension'
}
$run025Deadline = [DateTimeOffset]::UtcNow.AddMinutes(90)
$run025GuardScript = Join-Path $run025Run '06_billing_guard.ps1'
$run025Args = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$run025GuardScript`"",
    '-PodId', $run025Lease.pod_id,
    '-ExpectedName', $run025Lease.name,
    '-DeadlineUtc', $run025Deadline.ToString('o'),
    '-RunpodctlPath', "`"$run025Cli`"",
    '-LogPath', "`"$run025Log`""
)
$run025NewGuard = Start-Process powershell.exe `
    -WindowStyle Hidden `
    -ArgumentList $run025Args `
    -PassThru `
    -RedirectStandardError $run025ErrorLog
$run025ArmDeadline = [DateTimeOffset]::UtcNow.AddSeconds(20)
while ([DateTimeOffset]::UtcNow -lt $run025ArmDeadline) {
    Start-Sleep -Milliseconds 250
    if ($run025NewGuard.HasExited) { throw 'Replacement guard exited before arming' }
    if ((Get-Content -LiteralPath $run025Log -Raw -ErrorAction SilentlyContinue) -match ' ARMED ') {
        break
    }
}
if (-not ((Get-Content -LiteralPath $run025Log -Raw -ErrorAction SilentlyContinue) -match ' ARMED ')) {
    Stop-Process -Id $run025NewGuard.Id -ErrorAction SilentlyContinue
    throw 'Replacement guard did not confirm arming'
}
Stop-Process -Id $run025OldGuard.Id -ErrorAction Stop
$run025Record = [ordered]@{
    created_utc = [DateTimeOffset]::UtcNow.ToString('o')
    prior_deadline_utc = $run025Prior.replacement_deadline_utc
    replacement_deadline_utc = $run025Deadline.ToString('o')
    replacement_guard_pid = $run025NewGuard.Id
    prior_guard_retired = $true
    added_minutes_max = 90
    hourly_usd = [double]$run025Lease.returned_cost_per_hour
    added_gpu_cost_usd_max = [Math]::Round(1.5 * [double]$run025Lease.returned_cost_per_hour, 2)
    purpose = 'Run one bounded exploratory 70M development search, retrieve all evidence, and tear down'
}
$run025Record | ConvertTo-Json | Set-Content -LiteralPath $run025RecordPath -Encoding UTF8
[ordered]@{
    extension_armed = $true
    prior_guard_retired = $true
    replacement_deadline_utc = $run025Deadline.ToString('o')
    added_gpu_cost_usd_max = $run025Record.added_gpu_cost_usd_max
} | ConvertTo-Json -Compress
