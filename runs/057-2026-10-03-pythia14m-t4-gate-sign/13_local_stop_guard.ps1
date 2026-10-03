param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run057-*') { throw 'Invalid scoped identity' }
$run057Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run057Raw = & $run057Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run057Target = $run057Raw | ConvertFrom-Json
 if ($run057Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run057Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run057Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run057Seconds
}
for ($run057Try=1; $run057Try -le 5; $run057Try++) {
 try {
  $null = Read-Target
  & $run057Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run057After = Read-Target
  if ($run057After.status -notin @('EXITED','STOPPED') -and $run057After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run057Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
