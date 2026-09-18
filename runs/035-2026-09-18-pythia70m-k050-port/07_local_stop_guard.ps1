param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run035-*') { throw 'Invalid scoped identity' }
$run035Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run035Raw = & $run035Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run035Target = $run035Raw | ConvertFrom-Json
 if ($run035Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run035Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run035Remaining = ($DeadlineUtc - [DateTimeOffset]::UtcNow).TotalSeconds
 Start-Sleep -Seconds ([Math]::Min(30, [Math]::Max(1, [Math]::Ceiling($run035Remaining))))
}
for ($run035Try=1; $run035Try -le 5; $run035Try++) {
 try {
  $null = Read-Target
  & $run035Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run035After = Read-Target
  if ($run035After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run035Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1
