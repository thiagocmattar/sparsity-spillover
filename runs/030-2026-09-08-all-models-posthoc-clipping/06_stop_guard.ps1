param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run030-*') { throw 'Invalid scoped identity' }
$run030Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run030Raw = & $run030Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run030Target = $run030Raw | ConvertFrom-Json
 if ($run030Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run030Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) { Start-Sleep -Seconds 60 }
for ($run030Try=1; $run030Try -le 5; $run030Try++) {
 try {
  $null = Read-Target
  & $run030Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run030After = Read-Target
  if ($run030After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run030Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
