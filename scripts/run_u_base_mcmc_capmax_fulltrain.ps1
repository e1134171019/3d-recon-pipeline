param(
    [int]$CapMax = 750000,
    [switch]$Antialiased
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Label = if ($CapMax -ge 1000000) { "1m" } else { "{0}k" -f [int]($CapMax / 1000) }
$ModeSuffix = if ($Antialiased) { "_aa" } else { "" }
$RunRoot = Join-Path $Root "outputs\experiments\train_probes\u_base_mcmc_capmax${ModeSuffix}_fulltrain_$Timestamp"

$ImgDir = Join-Path $Root "data\frames_1600"
$Sparse = Join-Path $Root "outputs\SfM_models\sift\sparse\0"
$Report = Join-Path $Root "outputs\reports\pointcloud_validation_report.json"
$Out = Join-Path $RunRoot "mcmc_capmax_${Label}${ModeSuffix}"
$Log = Join-Path $RunRoot "logs\mcmc_capmax_${Label}${ModeSuffix}.log"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

Write-Host "Starting U_base MCMC cap_max full train: $Label ($CapMax)"
Write-Host "run root : $RunRoot"
Write-Host "imgdir   : $ImgDir"
Write-Host "sparse   : $Sparse"
Write-Host "report   : $Report"
Write-Host "steps    : 30000"
Write-Host "aa       : $(if ($Antialiased) { 'ON' } else { 'OFF' })"
Write-Host ""

$Cmd = @(
    $Python, "-u", "-m", "src.train_3dgs",
    "--train-mode", "mcmc",
    "--imgdir", $ImgDir,
    "--colmap", $Sparse,
    "--outdir", $Out,
    "--iterations", "30000",
    "--eval-steps", "1000",
    "--validation-report", $Report,
    "--cap-max", "$CapMax"
)

if ($Antialiased) {
    $Cmd += "--antialiased"
}

& $Python $RunAndTee --log $Log -- @Cmd

if ($LASTEXITCODE -ne 0) {
    throw "MCMC cap_max full train failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[OK] U_base MCMC cap_max full train finished: $RunRoot"
