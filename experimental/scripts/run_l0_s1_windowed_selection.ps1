param(
    [string]$ImgDir = "C:\3d-recon-pipeline\data\frames_1600",
    [int]$WindowSize = 6,
    [int]$KeepTopK = 1,
    [int]$MinPoints3d = 1
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$teeScript = Join-Path $repoRoot "scripts\run_and_tee.py"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $repoRoot "outputs\experiments\l0_selection\L0_S1_windowed_selection_$timestamp"
$logPath = Join-Path $runRoot "logs\route.log"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null

Write-Host "L0-S1 Windowed Frame Selection"
Write-Host "ImgDir      : $ImgDir"
Write-Host "WindowSize  : $WindowSize"
Write-Host "KeepTopK    : $KeepTopK"
Write-Host "MinPoints3d : $MinPoints3d"
Write-Host "RunRoot     : $runRoot"
Write-Host ""

$cmd = @(
    $pythonExe,
    "-u",
    "-m",
    "src.run_l0_windowed_selection",
    "--imgdir", $ImgDir,
    "--run-root", $runRoot,
    "--window-size", "$WindowSize",
    "--keep-top-k", "$KeepTopK",
    "--min-points3d", "$MinPoints3d"
)

& $pythonExe $teeScript --log $logPath -- $cmd
exit $LASTEXITCODE
