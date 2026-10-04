param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run060-*') { throw 'Invalid scoped identity' }
$run060Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run060Raw = & $run060Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run060Target = $run060Raw | ConvertFrom-Json
 if ($run060Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run060Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run060Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run060Seconds
}
for ($run060Try=1; $run060Try -le 5; $run060Try++) {
 try {
  $null = Read-Target
  & $run060Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run060After = Read-Target
  if ($run060After.status -notin @('EXITED','STOPPED') -and $run060After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run060Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
