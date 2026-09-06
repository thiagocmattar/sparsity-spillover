param()
$ErrorActionPreference='Stop'
$run028Root=(Resolve-Path -LiteralPath "$PSScriptRoot/../..").Path
$run028Control=Join-Path $PSScriptRoot 'launch-control/rtx5090-001'
$run028Previous=Get-Content -LiteralPath (Join-Path $run028Control 'lease-extension-001.json') -Raw | ConvertFrom-Json
$run028Receipt=Join-Path $run028Control 'lease-extension-002.json'
if (Test-Path -LiteralPath $run028Receipt) { throw 'Extension identity already exists' }
if ($run028Previous.pod_id -ne 'r7ex2sb18ax2yp' -or $run028Previous.guard_pid -ne 32840) { throw 'Unexpected prior lease' }
$run028Cli=Join-Path $run028Root 'tmp/runpodctl-v2.12.0.exe'
$run028Target=(& $run028Cli pod get $run028Previous.pod_id) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $run028Target.name -ne 'run028-rtx5090-001' -or $run028Target.costPerHr -gt 0.69) { throw 'Pod identity/price changed' }
$run028Old=Get-CimInstance Win32_Process -Filter 'ProcessId=32840'
if (-not $run028Old -or $run028Old.CommandLine -notlike '*04_stop_guard.ps1*' -or $run028Old.CommandLine -notlike '*r7ex2sb18ax2yp*') { throw 'Old guard ownership not verified' }
$run028Deadline='2026-09-07T00:57:06.1038427+00:00'
$run028Log=Join-Path $run028Control 'guard-extension-002.log'
$run028Args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSScriptRoot/04_stop_guard.ps1`"",'-PodId',$run028Previous.pod_id,'-ExpectedName',$run028Previous.name,'-DeadlineUtc',$run028Deadline,'-RunpodctlPath',"`"$run028Cli`"",'-LogPath',"`"$run028Log`"")
$run028New=Start-Process powershell.exe -ArgumentList $run028Args -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 4
$run028NewProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$($run028New.Id)"
if (-not $run028NewProcess -or -not (Test-Path -LiteralPath $run028Log) -or -not (Select-String -LiteralPath $run028Log -SimpleMatch 'ARMED r7ex2sb18ax2yp' -Quiet)) { throw 'New guard not armed; old guard retained' }
Stop-Process -Id 32840 -PassThru | Wait-Process -Timeout 10
if (Get-CimInstance Win32_Process -Filter 'ProcessId=32840') { throw 'Old guard still alive' }
$run028Result=[ordered]@{pod_id=$run028Previous.pod_id;name=$run028Previous.name;extension_utc=[DateTimeOffset]::UtcNow.ToString('o');previous_guard_pid=32840;guard_pid=$run028New.Id;stop_deadline_utc=$run028Deadline;original_start_utc=$run028Previous.original_start_utc;max_total_hours=8;returned_rate=$run028Target.costPerHr;max_gpu_cost_usd=5.52;total_run_budget_usd=20;storage='Existing 40 GB container and 40 GB Pod volume; pre-existing network volume unchanged';reason='Bounded continued kernel development plus time for a possible new 105-process final study; stop earlier after verified retrieval'}
$run028Result | ConvertTo-Json | Set-Content -LiteralPath $run028Receipt -Encoding UTF8
$run028Result | ConvertTo-Json -Compress
