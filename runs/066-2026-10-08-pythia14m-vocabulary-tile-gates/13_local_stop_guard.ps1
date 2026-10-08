param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run066-*') { throw 'Invalid scoped identity' }
$run066Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run066Raw = & $run066Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run066Target = $run066Raw | ConvertFrom-Json
 if ($run066Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run066Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run066Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run066Seconds
}
for ($run066Try=1; $run066Try -le 5; $run066Try++) {
 try {
  $null = Read-Target
  & $run066Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run066After = Read-Target
  if ($run066After.status -notin @('EXITED','STOPPED') -and $run066After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run066Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
