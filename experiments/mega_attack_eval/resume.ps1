$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$runner = Join-Path $taskRoot 'run.py'
$results = Join-Path $taskRoot 'results_v1'
$python = 'C:\habib dissertation\Projects\AML-GNN-Replication-v2\.venv\Scripts\python.exe'
$live = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'mega_attack_eval[\\/]+run\.py' })
if ($live.Count -gt 0) { throw 'A MEGA attack worker is already running. No second worker started.' }
if (Test-Path -LiteralPath (Join-Path $results 'status.json')) {
    $state = Get-Content -LiteralPath (Join-Path $results 'status.json') -Raw | ConvertFrom-Json
    if ($state.state -eq 'complete') { throw 'Evaluation is already complete. Verify and report; do not rerun.' }
}
if (Test-Path -LiteralPath (Join-Path $results 'protocol.json')) {
    $protocol = Get-Content -LiteralPath (Join-Path $results 'protocol.json') -Raw | ConvertFrom-Json
    foreach ($entry in $protocol.source_hashes.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLower() -ne $entry.Value) { throw "Frozen source/input changed: $($entry.Name)" }
    }
}
Set-Location -LiteralPath $taskRoot
$ErrorActionPreference = 'Continue'
& $python -B -u $runner >> (Join-Path $taskRoot 'full_run.log') 2>> (Join-Path $taskRoot 'full_run_errors.log')
$workerExit = $LASTEXITCODE
[ordered]@{timestamp=(Get-Date -Format o); exit_code=$workerExit} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'last_exit.json') -Encoding UTF8
exit $workerExit
