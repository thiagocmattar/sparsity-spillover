param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run044-*') { throw 'Invalid scoped identity' }
$run044Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run044Raw = & $run044Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run044Target = $run044Raw | ConvertFrom-Json
 if ($run044Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run044Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run044Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run044Seconds
}
for ($run044Try=1; $run044Try -le 5; $run044Try++) {
 try {
  $null = Read-Target
  & $run044Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run044After = Read-Target
  if ($run044After.status -notin @('EXITED','STOPPED') -and $run044After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run044Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
