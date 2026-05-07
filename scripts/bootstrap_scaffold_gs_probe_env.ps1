param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$PythonExe = "C:\3d-recon-pipeline\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$venvRoot = Join-Path $SandboxRoot ".venv_scaffold"
if (-not (Test-Path -LiteralPath $venvRoot)) {
    & $PythonExe -m venv $venvRoot
}

$venvPython = Join-Path $venvRoot "Scripts\python.exe"
Write-Host "[OK] Scaffold-GS sandbox venv ready"
Write-Host "Python: $venvPython"
Write-Host ""
Write-Host "[NEXT] Install minimal deps into sandbox venv:"
Write-Host "       pip install --upgrade pip setuptools wheel"
Write-Host "       pip install einops plyfile wandb"
Write-Host ""
Write-Host "[NEXT] Initialize Scaffold-GS submodules before build/runtime."
