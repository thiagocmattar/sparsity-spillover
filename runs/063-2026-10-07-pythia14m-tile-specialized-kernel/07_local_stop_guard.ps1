param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run063-*') { throw 'Invalid scoped identity' }
$run063Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run063Raw = & $run063Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run063Target = $run063Raw | ConvertFrom-Json
 if ($run063Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run063Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run063Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run063Seconds
}
for ($run063Try=1; $run063Try -le 5; $run063Try++) {
 try {
  $null = Read-Target
  & $run063Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run063After = Read-Target
  if ($run063After.status -notin @('EXITED','STOPPED') -and $run063After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run063Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
