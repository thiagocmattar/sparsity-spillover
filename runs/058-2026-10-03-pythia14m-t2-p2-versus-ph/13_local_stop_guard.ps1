param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run058-*') { throw 'Invalid scoped identity' }
$run058Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run058Raw = & $run058Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run058Target = $run058Raw | ConvertFrom-Json
 if ($run058Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run058Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run058Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run058Seconds
}
for ($run058Try=1; $run058Try -le 5; $run058Try++) {
 try {
  $null = Read-Target
  & $run058Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run058After = Read-Target
  if ($run058After.status -notin @('EXITED','STOPPED') -and $run058After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run058Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
