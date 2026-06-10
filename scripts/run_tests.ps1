param(
    [string[]]$PytestArgs = @("tests")
)

$ErrorActionPreference = "Stop"

$Python = if ($env:DIAAD_PYTHON) {
    $env:DIAAD_PYTHON
} else {
    "$env:USERPROFILE\anaconda3\envs\diaad\python.exe"
}

if (-not (Test-Path $Python)) {
    throw "Could not find DIAAD Python at $Python. Set DIAAD_PYTHON to override."
}

& $Python -m pytest @PytestArgs
exit $LASTEXITCODE
