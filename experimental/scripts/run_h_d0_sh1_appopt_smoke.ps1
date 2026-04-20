$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\train_probes\sh1_appopt_h_d0_smoke_$Timestamp"

$ImgDir = Join-Path $Root "outputs\experiments\l0_selection_compare\l0_2x2_20260418_020711\H_D0\selected_frames"
$Sparse = Join-Path $Root "outputs\experiments\l0_selection_compare\l0_2x2_20260418_020711\H_D0\SfM_models\l0_s1_windowed\sparse\1"
$Report = Join-Path $Root "outputs\experiments\l0_selection_compare\l0_2x2_20260418_020711\H_D0\reports\pointcloud_validation_report.json"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host "Starting H_D0 sh_degree=1 + app_opt smoke: $RunRoot"
Write-Host "imgdir  : $ImgDir"
Write-Host "sparse  : $Sparse"
Write-Host "report  : $Report"
Write-Host ""

$Out = Join-Path $RunRoot "h_d0_sh1_appopt_shorttrain"
$Log = Join-Path $RunRoot "logs\train.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

$Cmd = @(
    $Python, "-u", "-m", "src.train_3dgs",
    "--imgdir", $ImgDir,
    "--colmap", $Sparse,
    "--outdir", $Out,
    "--iterations", "5000",
    "--eval-steps", "1000",
    "--grow-grad2d", "0.0008",
    "--validation-report", $Report,
    "--disable-video",
    "--sh-degree", "1",
    "--app-opt"
)

& $Python $RunAndTee --log $Log -- @Cmd

Write-Host ""
Write-Host "[OK] H_D0 sh_degree=1 + app_opt smoke finished: $RunRoot"
