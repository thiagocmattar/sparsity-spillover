param([string]$Attempt='rtx5090-001',[string]$Gpu='NVIDIA GeForce RTX 5090',[double]$QuotedRate=0.69,[double]$Hours=4,[ValidateSet('COMMUNITY','SECURE')][string]$CloudType='COMMUNITY',[string]$Country='CA')
$ErrorActionPreference='Stop'
if ($Attempt -notmatch '^[a-z0-9-]+$' -or $Hours -le 0 -or $Hours -gt 4 -or $QuotedRate -le 0 -or $QuotedRate*$Hours -gt 14) { throw 'Invalid scoped lease' }
$run028Root=(Resolve-Path "$PSScriptRoot/../..").Path
$run028Cli=Join-Path $run028Root 'tmp/runpodctl-v2.12.0.exe'
$run028Folder=Join-Path $PSScriptRoot "launch-control/$Attempt"
$run028Name="run028-$Attempt"
if (Test-Path -LiteralPath $run028Folder) { throw 'Attempt folder exists; inspect resources before retry' }
$null=New-Item -ItemType Directory -Path $run028Folder
$run028Start=[DateTimeOffset]::UtcNow
$run028Deadline=$run028Start.AddHours($Hours)
$run028Placement=@()
if ($Country) { $run028Placement=@('--country-code',$Country) }
$run028Raw=& $run028Cli pod create --name $run028Name --image 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35' --cloud-type $CloudType --gpu-id $Gpu --gpu-count 1 --min-cuda-version 12.8 --container-disk-in-gb 40 --volume-in-gb 40 --volume-mount-path /workspace --ports '22/tcp' --public-ip --ssh @run028Placement
if ($LASTEXITCODE -ne 0) { throw 'Create failed: list resources by exact name before retry' }
$run028Pod=$run028Raw | ConvertFrom-Json
if (-not $run028Pod.id -or $run028Pod.name -ne $run028Name) { throw 'Unexpected response: inspect control plane' }
$run028Args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSScriptRoot/04_stop_guard.ps1`"",'-PodId',$run028Pod.id,'-ExpectedName',$run028Name,'-DeadlineUtc',$run028Deadline.ToString('o'),'-RunpodctlPath',"`"$run028Cli`"",'-LogPath',"`"$run028Folder/guard.log`"")
$run028Guard=Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $run028Args -PassThru -RedirectStandardError "$run028Folder/guard.err"
$run028Lease=[ordered]@{pod_id=$run028Pod.id;name=$run028Name;cloud_type=$CloudType;start_utc=$run028Start.ToString('o');stop_deadline_utc=$run028Deadline.ToString('o');quoted_rate=$QuotedRate;returned_rate=$run028Pod.costPerHr;max_hours=$Hours;guard_pid=$run028Guard.Id;total_run_budget_usd=20;workspace='/workspace/sparsity-spillover';image='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'}
$run028Lease | ConvertTo-Json | Set-Content -LiteralPath "$run028Folder/lease.json" -Encoding UTF8
$run028Lease | ConvertTo-Json
