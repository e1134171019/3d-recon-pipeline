$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\train_probes\u_base_mcmc_capmax_mid_probe_$Timestamp"

$ImgDir = Join-Path $Root "data\frames_1600"
$Sparse = Join-Path $Root "outputs\SfM_models\sift\sparse\0"
$Report = Join-Path $Root "outputs\reports\pointcloud_validation_report.json"

function Invoke-CapMaxProbe {
    param(
        [Parameter(Mandatory = $true)][int]$CapMax,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $Out = Join-Path $RunRoot "mcmc_capmax_$Label"
    $Log = Join-Path $RunRoot "logs\mcmc_capmax_$Label.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

    Write-Host "Starting U_base MCMC cap_max probe: $Label ($CapMax)"
    Write-Host "run root : $RunRoot"
    Write-Host "imgdir   : $ImgDir"
    Write-Host "sparse   : $Sparse"
    Write-Host "report   : $Report"
    Write-Host "steps    : 7000"
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
        "--cap-max", "$CapMax",
        "--disable-video"
    )

    & $Python $RunAndTee --log $Log -- @Cmd

    Write-Host ""
    Write-Host "[OK] cap_max probe finished: $Label"
    Write-Host ""
}

Invoke-CapMaxProbe -CapMax 750000 -Label "750k"
Invoke-CapMaxProbe -CapMax 500000 -Label "500k"

Write-Host "[OK] All U_base MCMC cap_max mid probes finished: $RunRoot"
