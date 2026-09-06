param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference='Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run028-*') { throw 'Invalid scoped identity' }
$run028Cli=(Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run028Raw=& $run028Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run028Target=$run028Raw | ConvertFrom-Json
 if ($run028Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run028Target
}
$null=Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) { Start-Sleep -Seconds 60 }
for ($run028Try=1; $run028Try -le 5; $run028Try++) {
 try {
  $null=Read-Target
  & $run028Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run028After=Read-Target
  if ($run028After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run028Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
