param([string]$Python = 'D:/Huawei Cup/.venv/Scripts/python.exe')
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath 'outputs/frozen_manifest.json')) {
        throw 'Original A-D experiment is required. Run run.ps1 -Stage all first.'
    }
    if (-not (Test-Path -LiteralPath 'outputs/expanded/frozen_manifest.json')) {
        & $Python har_rf.py train
        if ($LASTEXITCODE -ne 0) { throw 'RF training failed' }
    }
    if (-not (Test-Path -LiteralPath 'outputs/expanded/results.json')) {
        & $Python har_rf.py evaluate
        if ($LASTEXITCODE -ne 0) { throw 'RF evaluation failed' }
    }
    & $Python har_rf.py validate
    if ($LASTEXITCODE -ne 0) { throw 'Model validation failed' }
    & $Python make_figures.py --results-dir outputs/expanded
    if ($LASTEXITCODE -ne 0) { throw 'Plotting failed' }
    & $Python analyze_expanded.py --results-dir outputs/expanded
    if ($LASTEXITCODE -ne 0) { throw 'Descriptive analysis failed' }
} finally { Pop-Location }
