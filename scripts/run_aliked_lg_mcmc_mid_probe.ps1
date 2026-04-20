$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\train_probes\aliked_lg_mcmc_mid_probe_$Timestamp"

$ImgDir = Join-Path $Root "data\frames_1600"
$Base = Join-Path $Root "outputs\experiments\route3_hloc_aliked_lg_20260411_170852"
$Sparse = Join-Path $Base "SfM_models\hloc_aliked_n16_aliked_lightglue\sparse\0"
$Report = Join-Path $Base "reports\pointcloud_validation_report.json"
$Out = Join-Path $RunRoot "aliked_lg_mcmc"
$Log = Join-Path $RunRoot "logs\aliked_lg_mcmc.log"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

Write-Host "Starting ALIKED + LightGlue + MCMC mid probe: $RunRoot"
Write-Host "imgdir  : $ImgDir"
Write-Host "sparse  : $Sparse"
Write-Host "report  : $Report"
Write-Host "steps   : 7000"
Write-Host ""

$Cmd = @(
    $Python, "-u", "-m", "src.train_3dgs",
    "--train-mode", "mcmc",
    "--imgdir", $ImgDir,
    "--colmap", $Sparse,
    "--outdir", $Out,
    "--iterations", "7000",
    "--eval-steps", "1000",
    "--validation-report", $Report,
    "--disable-video"
)

& $Python $RunAndTee --log $Log -- @Cmd

Write-Host ""
Write-Host "[OK] ALIKED + LightGlue + MCMC mid probe finished: $RunRoot"
