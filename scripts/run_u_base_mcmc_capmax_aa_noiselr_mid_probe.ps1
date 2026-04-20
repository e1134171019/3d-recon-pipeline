$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\train_probes\u_base_mcmc_capmax_aa_noiselr_mid_probe_$Timestamp"

$ImgDir = Join-Path $Root "data\frames_1600"
$Sparse = Join-Path $Root "outputs\SfM_models\sift\sparse\0"
$Report = Join-Path $Root "outputs\reports\pointcloud_validation_report.json"
$Out = Join-Path $RunRoot "mcmc_capmax_750k_aa_noiselr_100k"
$Log = Join-Path $RunRoot "logs\mcmc_capmax_750k_aa_noiselr_100k.log"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

Write-Host "Starting U_base MCMC cap_max + antialiased + noise_lr probe"
Write-Host "run root : $RunRoot"
Write-Host "imgdir   : $ImgDir"
Write-Host "sparse   : $Sparse"
Write-Host "report   : $Report"
Write-Host "steps    : 7000"
Write-Host "cap_max  : 750000"
Write-Host "aa       : ON"
Write-Host "noise_lr : 100000"
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
    "--cap-max", "750000",
    "--antialiased",
    "--mcmc-noise-lr", "100000",
    "--disable-video"
)

& $Python $RunAndTee --log $Log -- @Cmd

if ($LASTEXITCODE -ne 0) {
    throw "MCMC cap_max + antialiased + noise_lr probe failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[OK] U_base MCMC cap_max + antialiased + noise_lr probe finished: $RunRoot"
