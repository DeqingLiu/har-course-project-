param(
    [string]$Python = 'D:/Huawei Cup/.venv/Scripts/python.exe',
    [ValidateSet('all','prepare','train','evaluate','validate','figures')]
    [string]$Stage = 'all'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $PSScriptRoot
try {
    if ((Test-Path -LiteralPath 'outputs/expanded/frozen_manifest.json') -and ($Stage -ne 'figures')) {
        throw 'A-D evidence is referenced by the frozen RF extension. Use run_expanded.ps1 to verify the complete experiment; use a fresh source directory for retraining.'
    }
    $steps = if ($Stage -eq 'all') { @('prepare','train','evaluate','validate','figures') } else { @($Stage) }
    foreach ($step in $steps) {
        if (($step -eq 'train') -and (Test-Path -LiteralPath 'outputs/frozen_manifest.json')) {
            Write-Host 'Existing sealed experiment: skipping retraining. Use a fresh directory for a new run.'
            continue
        }
        if (($step -eq 'prepare') -and (Test-Path -LiteralPath 'outputs/frozen_manifest.json')) {
            Write-Host 'Existing sealed data: skipping preparation.'
            continue
        }
        if ($step -eq 'figures') { & $Python make_figures.py } else { & $Python har.py $step }
        if ($LASTEXITCODE -ne 0) { throw "Stage failed: $step" }
    }
} finally { Pop-Location }
