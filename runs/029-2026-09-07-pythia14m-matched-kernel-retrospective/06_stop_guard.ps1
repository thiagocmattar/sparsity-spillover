param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run029-*') { throw 'Invalid scoped identity' }
$run029Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run029Raw = & $run029Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run029Target = $run029Raw | ConvertFrom-Json
 if ($run029Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run029Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) { Start-Sleep -Seconds 60 }
for ($run029Try=1; $run029Try -le 5; $run029Try++) {
 try {
  $null = Read-Target
  & $run029Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run029After = Read-Target
  if ($run029After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run029Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
