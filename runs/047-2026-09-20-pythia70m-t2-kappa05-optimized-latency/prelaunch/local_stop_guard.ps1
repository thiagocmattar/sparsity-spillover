param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run047-*') { throw 'Invalid scoped identity' }
$run047Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run047Raw = & $run047Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run047Target = $run047Raw | ConvertFrom-Json
 if ($run047Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run047Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run047Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run047Seconds
}
for ($run047Try=1; $run047Try -le 5; $run047Try++) {
 try {
  $null = Read-Target
  & $run047Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run047After = Read-Target
  if ($run047After.status -notin @('EXITED','STOPPED') -and $run047After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run047Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
