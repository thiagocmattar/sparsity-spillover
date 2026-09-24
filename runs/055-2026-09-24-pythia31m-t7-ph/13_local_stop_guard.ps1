param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run055-*') { throw 'Invalid scoped identity' }
$run055Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run055Raw = & $run055Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run055Target = $run055Raw | ConvertFrom-Json
 if ($run055Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run055Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run055Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run055Seconds
}
for ($run055Try=1; $run055Try -le 5; $run055Try++) {
 try {
  $null = Read-Target
  $null = & $run055Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run055After = Read-Target
  if ($run055After.status -notin @('EXITED','STOPPED') -and $run055After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run055Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
