param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run067-*') { throw 'Invalid scoped identity' }
$run067Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run067Raw = & $run067Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run067Target = $run067Raw | ConvertFrom-Json
 if ($run067Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run067Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run067Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run067Seconds
}
for ($run067Try=1; $run067Try -le 5; $run067Try++) {
 try {
  $null = Read-Target
  & $run067Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run067After = Read-Target
  if ($run067After.status -notin @('EXITED','STOPPED') -and $run067After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run067Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
