param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run042-*') { throw 'Invalid scoped identity' }
$run042Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run042Raw = & $run042Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run042Target = $run042Raw | ConvertFrom-Json
 if ($run042Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run042Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run042Remaining = ($DeadlineUtc - [DateTimeOffset]::UtcNow).TotalSeconds
 Start-Sleep -Seconds ([Math]::Min(30, [Math]::Max(1, [Math]::Ceiling($run042Remaining))))
}
for ($run042Try=1; $run042Try -le 5; $run042Try++) {
 try {
  $null = Read-Target
  & $run042Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run042After = Read-Target
  if ($run042After.status -notin @('EXITED','STOPPED') -and
      $run042After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run042Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
