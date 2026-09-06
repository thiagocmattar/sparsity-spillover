param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run027-*') { throw 'Invalid scoped identity' }
$run027Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run027Raw = & $run027Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run027Target = $run027Raw | ConvertFrom-Json
 if ($run027Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run027Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) { Start-Sleep -Seconds 60 }
for ($run027Try=1; $run027Try -le 5; $run027Try++) {
 try {
  $null = Read-Target
  & $run027Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run027After = Read-Target
  if ($run027After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run027Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
