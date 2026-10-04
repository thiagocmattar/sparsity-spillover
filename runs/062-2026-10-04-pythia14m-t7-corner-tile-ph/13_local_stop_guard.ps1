param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run062-*') { throw 'Invalid scoped identity' }
$run062Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run062Raw = & $run062Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run062Target = $run062Raw | ConvertFrom-Json
 if ($run062Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run062Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run062Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run062Seconds
}
for ($run062Try=1; $run062Try -le 5; $run062Try++) {
 try {
  $null = Read-Target
  & $run062Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run062After = Read-Target
  if ($run062After.status -notin @('EXITED','STOPPED') -and $run062After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run062Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
