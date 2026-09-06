param()
$ErrorActionPreference='Stop'
$run028Root=(Resolve-Path -LiteralPath "$PSScriptRoot/../..").Path
$run028Control=Join-Path $PSScriptRoot 'launch-control/rtx5090-001'
$run028Lease=Get-Content -LiteralPath (Join-Path $run028Control 'lease.json') -Raw | ConvertFrom-Json
$run028Receipt=Join-Path $run028Control 'lease-extension-001.json'
if (Test-Path -LiteralPath $run028Receipt) { throw 'Extension identity already exists' }
if ($run028Lease.pod_id -ne 'r7ex2sb18ax2yp' -or $run028Lease.guard_pid -ne 21920) { throw 'Unexpected original lease' }
$run028Cli=Join-Path $run028Root 'tmp/runpodctl-v2.12.0.exe'
$run028Target=(& $run028Cli pod get $run028Lease.pod_id) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $run028Target.name -ne 'run028-rtx5090-001' -or $run028Target.costPerHr -gt 0.69) { throw 'Pod identity/price changed' }
$run028Old=Get-CimInstance Win32_Process -Filter 'ProcessId=21920'
if (-not $run028Old -or $run028Old.CommandLine -notlike '*04_stop_guard.ps1*' -or $run028Old.CommandLine -notlike '*r7ex2sb18ax2yp*') { throw 'Old guard ownership not verified' }
$run028Deadline='2026-09-06T22:57:06.1038427+00:00'
$run028Log=Join-Path $run028Control 'guard-extension-001.log'
$run028Args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSScriptRoot/04_stop_guard.ps1`"",'-PodId',$run028Lease.pod_id,'-ExpectedName',$run028Lease.name,'-DeadlineUtc',$run028Deadline,'-RunpodctlPath',"`"$run028Cli`"",'-LogPath',"`"$run028Log`"")
$run028New=Start-Process powershell.exe -ArgumentList $run028Args -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 4
$run028NewProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$($run028New.Id)"
if (-not $run028NewProcess -or -not (Test-Path -LiteralPath $run028Log) -or -not (Select-String -LiteralPath $run028Log -SimpleMatch 'ARMED r7ex2sb18ax2yp' -Quiet)) { throw 'New guard not armed; old guard retained' }
Stop-Process -Id 21920
if (Get-Process -Id 21920 -ErrorAction SilentlyContinue) { throw 'Old guard still alive' }
$run028Result=[ordered]@{pod_id=$run028Lease.pod_id;name=$run028Lease.name;extension_utc=[DateTimeOffset]::UtcNow.ToString('o');original_guard_pid=21920;guard_pid=$run028New.Id;stop_deadline_utc=$run028Deadline;original_start_utc=$run028Lease.start_utc;max_total_hours=6;returned_rate=$run028Target.costPerHr;max_gpu_cost_usd=4.14;total_run_budget_usd=20;storage='Existing 40 GB container and 40 GB Pod volume; pre-existing network volume unchanged';reason='Sequential K037/K038 development and possible qualified follow-up matrix; stop earlier after verified retrieval'}
$run028Result | ConvertTo-Json | Set-Content -LiteralPath $run028Receipt -Encoding UTF8
$run028Result | ConvertTo-Json -Compress
