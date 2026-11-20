$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    & $venvPy (Join-Path $PSScriptRoot "dev.py") @args
} else {
    & py (Join-Path $PSScriptRoot "dev.py") @args
}
exit $LASTEXITCODE
