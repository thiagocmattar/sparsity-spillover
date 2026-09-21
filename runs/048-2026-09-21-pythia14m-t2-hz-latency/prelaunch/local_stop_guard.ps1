param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run048-*') { throw 'Invalid scoped identity' }
$run048Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run048Raw = & $run048Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run048Target = $run048Raw | ConvertFrom-Json
 if ($run048Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run048Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run048Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run048Seconds
}
for ($run048Try=1; $run048Try -le 5; $run048Try++) {
 try {
  $null = Read-Target
  & $run048Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run048After = Read-Target
  if ($run048After.status -notin @('EXITED','STOPPED') -and $run048After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run048Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
