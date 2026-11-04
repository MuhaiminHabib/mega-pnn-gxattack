$ErrorActionPreference = 'Stop'
$pilotRoot = $PSScriptRoot
$pilotRunner = Join-Path $pilotRoot 'run.py'
$pilotResults = Join-Path $pilotRoot 'results'
$pilotPython = 'C:\habib dissertation\Projects\AML-GNN-Replication-v2\.venv\Scripts\python.exe'
$pilotLive = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'mega_(attack_audit_v1|selected_attack_pilot_v1|attack_eval|priority_repair|pg_refinement_v1)[\\/].*\.py'
})
if ($pilotLive.Count -gt 0) { throw 'An experiment worker is already alive. No additional GPU worker started.' }
if (Test-Path -LiteralPath (Join-Path $pilotResults 'failure.json')) { throw 'Recorded failure requires diagnosis before resumption.' }
if (Test-Path -LiteralPath (Join-Path $pilotResults 'runner_verification.json')) { throw 'Pilot is complete; verify/report without rerunning.' }
if (Test-Path -LiteralPath (Join-Path $pilotResults 'protocol.json')) {
    $pilotProtocol = Get-Content -LiteralPath (Join-Path $pilotResults 'protocol.json') -Raw | ConvertFrom-Json
    foreach ($entry in $pilotProtocol.source_hashes.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLower() -ne $entry.Value) { throw "Frozen source or input changed: $($entry.Name)" }
    }
}
Set-Location -LiteralPath $pilotRoot
$ErrorActionPreference = 'Continue'
& $pilotPython -B -u $pilotRunner >> (Join-Path $pilotRoot 'run.log') 2>> (Join-Path $pilotRoot 'errors.log')
$pilotExit = $LASTEXITCODE
[ordered]@{timestamp=(Get-Date -Format o);exit_code=$pilotExit} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pilotRoot 'last_exit.json') -Encoding UTF8
exit $pilotExit
