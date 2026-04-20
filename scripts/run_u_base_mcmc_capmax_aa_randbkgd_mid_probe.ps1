$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\train_probes\u_base_mcmc_capmax_aa_randbkgd_mid_probe_$Timestamp"

$ImgDir = Join-Path $Root "data\frames_1600"
$Sparse = Join-Path $Root "outputs\SfM_models\sift\sparse\0"
$Report = Join-Path $Root "outputs\reports\pointcloud_validation_report.json"
$Out = Join-Path $RunRoot "mcmc_capmax_750k_aa_randbkgd"
$Log = Join-Path $RunRoot "logs\mcmc_capmax_750k_aa_randbkgd.log"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

Write-Host "Starting U_base MCMC cap_max + antialiased + random_bkgd probe"
Write-Host "run root : $RunRoot"
Write-Host "imgdir   : $ImgDir"
Write-Host "sparse   : $Sparse"
Write-Host "report   : $Report"
Write-Host "steps    : 7000"
Write-Host "cap_max  : 750000"
Write-Host "aa       : ON"
Write-Host "rnd_bkgd : ON"
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
    "--random-bkgd",
    "--disable-video"
)

& $Python $RunAndTee --log $Log -- @Cmd

if ($LASTEXITCODE -ne 0) {
    throw "MCMC cap_max + antialiased + random_bkgd probe failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[OK] U_base MCMC cap_max + antialiased + random_bkgd probe finished: $RunRoot"
