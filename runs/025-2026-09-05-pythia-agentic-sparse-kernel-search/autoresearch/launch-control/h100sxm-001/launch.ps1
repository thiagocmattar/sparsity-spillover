$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Name = 'run025-autoresearch-h100sxm-001'
$run025LeasePath = Join-Path $PSScriptRoot 'lease.json'
if (Test-Path -LiteralPath $run025LeasePath) {
    throw 'Lease already recorded; discover live resources instead of duplicating'
}
$run025Before = [DateTimeOffset]::UtcNow
$run025Deadline = $run025Before.AddHours(3)
$run025Raw = & $run025Cli pod create `
    --name $run025Name `
    --image 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35' `
    --cloud-type COMMUNITY `
    --gpu-id 'NVIDIA H100 80GB HBM3' `
    --gpu-count 1 `
    --min-cuda-version 12.8 `
    --container-disk-in-gb 40 `
    --volume-in-gb 60 `
    --volume-mount-path /workspace `
    --ports '22/tcp' `
    --public-ip `
    --ssh `
    --wait `
    --wait-timeout 5m
if ($LASTEXITCODE -ne 0) {
    throw 'Creation may have allocated a Pod; discover the exact name before retrying'
}
$run025Pod = $run025Raw | ConvertFrom-Json
if (-not $run025Pod.id -or $run025Pod.name -ne $run025Name) {
    throw 'Unexpected creation response; reconcile before doing anything else'
}
$run025Rate = [double]$run025Pod.costPerHr
if ($run025Rate -le 0 -or $run025Rate -gt 3.00) {
    & $run025Cli pod stop $run025Pod.id | Out-Null
    throw 'H100 hourly price exceeds the bounded launch definition; Pod stop requested'
}
$run025Guard = Join-Path $run025Run '06_billing_guard.ps1'
$run025Log = Join-Path $PSScriptRoot 'billing-guard.log'
$run025Args = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$run025Guard`"",
    '-PodId', $run025Pod.id,
    '-ExpectedName', $run025Name,
    '-DeadlineUtc', $run025Deadline.ToString('o'),
    '-RunpodctlPath', "`"$run025Cli`"",
    '-LogPath', "`"$run025Log`""
)
$run025Process = Start-Process powershell.exe `
    -WindowStyle Hidden `
    -ArgumentList $run025Args `
    -PassThru `
    -RedirectStandardError (Join-Path $PSScriptRoot 'guard-error.log')
$run025Lease = [ordered]@{
    pod_id = $run025Pod.id
    name = $run025Name
    requested_utc = $run025Before.ToString('o')
    stop_deadline_utc = $run025Deadline.ToString('o')
    quoted_gpu_usd_per_hour = $run025Rate
    lease_hours_max = 3
    maximum_gpu_cost_usd = [Math]::Round($run025Rate * 3, 4)
    study_total_usd_max = 40
    conservative_prior_debit_usd = 7.75
    guard_pid = $run025Process.Id
    image = 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'
    gpu = 'NVIDIA H100 80GB HBM3'
    cloud = 'COMMUNITY'
    ephemeral_volume_gb = 60
    purpose = 'fixed no-retuning H100 transfer: P0, frozen winners, repeat timing, graph controls, and component ablations'
    returned_cost_per_hour = $run025Pod.costPerHr
}
$run025Lease | ConvertTo-Json | Set-Content -LiteralPath $run025LeasePath -Encoding UTF8
$run025Lease | ConvertTo-Json
