param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run043-*') { throw 'Invalid scoped identity' }
$run043Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run043Raw = & $run043Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run043Target = $run043Raw | ConvertFrom-Json
 if ($run043Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run043Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run043Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run043Seconds
}
for ($run043Try=1; $run043Try -le 5; $run043Try++) {
 try {
  $null = Read-Target
  & $run043Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run043After = Read-Target
  if ($run043After.status -notin @('EXITED','STOPPED') -and $run043After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run043Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
