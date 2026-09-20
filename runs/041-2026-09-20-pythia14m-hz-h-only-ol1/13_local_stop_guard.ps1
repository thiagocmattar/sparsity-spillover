param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run041-*') { throw 'Invalid scoped identity' }
$run041Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run041Raw = & $run041Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run041Target = $run041Raw | ConvertFrom-Json
 if ($run041Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run041Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run041Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run041Seconds
}
for ($run041Try=1; $run041Try -le 5; $run041Try++) {
 try {
  $null = Read-Target
  & $run041Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run041After = Read-Target
  if ($run041After.status -notin @('EXITED','STOPPED') -and $run041After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run041Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
