param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run054-*') { throw 'Invalid scoped identity' }
$run054Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run054Raw = & $run054Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run054Target = $run054Raw | ConvertFrom-Json
 if ($run054Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run054Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run054Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run054Seconds
}
for ($run054Try=1; $run054Try -le 5; $run054Try++) {
 try {
  $null = Read-Target
  $null = & $run054Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run054After = Read-Target
  if ($run054After.status -notin @('EXITED','STOPPED') -and $run054After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run054Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
