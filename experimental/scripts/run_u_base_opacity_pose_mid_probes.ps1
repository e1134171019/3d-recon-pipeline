$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\train_probes\u_base_opacity_pose_mid_probes_$Timestamp"

$ImgDir = Join-Path $Root "data\frames_1600"
$Sparse = Join-Path $Root "outputs\SfM_models\sift\sparse\0"
$Report = Join-Path $Root "outputs\reports\pointcloud_validation_report.json"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host "Starting U_base opacity/pose mid probes: $RunRoot"
Write-Host "imgdir  : $ImgDir"
Write-Host "sparse  : $Sparse"
Write-Host "report  : $Report"
Write-Host "steps   : 7000"
Write-Host ""

function Invoke-Probe {
    param(
        [string]$Name,
        [string[]]$ExtraArgs
    )

    $Out = Join-Path $RunRoot $Name
    $Log = Join-Path $RunRoot "logs\$Name.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

    $Cmd = @(
        $Python, "-u", "-m", "src.train_3dgs",
        "--imgdir", $ImgDir,
        "--colmap", $Sparse,
        "--outdir", $Out,
        "--iterations", "7000",
        "--eval-steps", "1000",
        "--grow-grad2d", "0.0008",
        "--validation-report", $Report,
        "--disable-video"
    ) + $ExtraArgs

    Write-Host "[RUN] $Name"
    & $Python $RunAndTee --log $Log -- @Cmd
    Write-Host ""
}

Invoke-Probe -Name "opacity_reg_001" -ExtraArgs @("--opacity-reg", "0.01")
Invoke-Probe -Name "pose_opt" -ExtraArgs @("--pose-opt")

Write-Host "[OK] U_base opacity/pose mid probes finished: $RunRoot"
