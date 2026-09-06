param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run026-*') { throw 'Invalid scoped identity' }
$run026Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run026Raw = & $run026Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run026Target = $run026Raw | ConvertFrom-Json
 if ($run026Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run026Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) { Start-Sleep -Seconds 60 }
for ($run026Try=1; $run026Try -le 5; $run026Try++) {
 try {
  $null = Read-Target
  & $run026Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run026After = Read-Target
  if ($run026After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run026Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
