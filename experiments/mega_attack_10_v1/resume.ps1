$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = 'C:\habib dissertation\Projects\AML-GNN-Replication-v2\.venv\Scripts\python.exe'
$taskExisting = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'mega_attack_10_v1[\\/]supervise\.py'
})
if ($taskExisting.Count -gt 0) { throw 'The 10-node controller is already running; no duplicate started.' }
foreach ($name in @('controller_failure.json', 'worker_failure.json', 'results\failure.json')) {
    if (Test-Path -LiteralPath (Join-Path $taskRoot $name)) { throw "Recorded failure requires diagnosis: $name" }
}
if (Test-Path -LiteralPath (Join-Path $taskRoot 'results\independent_verification.json')) {
    throw 'The 10-node run is complete; report without rerunning.'
}
& $taskPython -B -u (Join-Path $taskRoot 'supervise.py') >> (Join-Path $taskRoot 'controller.log') 2>> (Join-Path $taskRoot 'controller_errors.log')
exit $LASTEXITCODE
