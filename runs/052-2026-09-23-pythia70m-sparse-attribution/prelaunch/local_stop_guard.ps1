param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run052-*') { throw 'Invalid scoped identity' }
$run052Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run052Raw = & $run052Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run052Target = $run052Raw | ConvertFrom-Json
 if ($run052Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run052Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run052Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run052Seconds
}
for ($run052Try=1; $run052Try -le 5; $run052Try++) {
 try {
  $null = Read-Target
  & $run052Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run052After = Read-Target
  if ($run052After.status -notin @('EXITED','STOPPED') -and $run052After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run052Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
