$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\l0_fast_validation\gate2_shorttrain_$Timestamp"

$BaselineImgDir = Join-Path $Root "outputs\experiments\l0_fast_validation\subsets\frames_1600_every6"
$BaselineSparse = Join-Path $Root "outputs\experiments\l0_fast_validation\gate1_subset_20260412_093904\baseline_subset\SfM_models\sift\sparse\0"
$BaselineReport = Join-Path $Root "outputs\experiments\l0_fast_validation\gate1_subset_20260412_093904\baseline_subset\reports\pointcloud_validation_report.json"

$L0ImgDir = Join-Path $Root "outputs\experiments\l0_selection\L0_S1_windowed_selection_20260412_100502\selected_frames"
$L0Sparse = Join-Path $Root "outputs\experiments\l0_selection\L0_S1_windowed_selection_20260412_100502\SfM_models\l0_s1_windowed\sparse\0"
$L0Report = Join-Path $Root "outputs\experiments\l0_selection\L0_S1_windowed_selection_20260412_100502\reports\pointcloud_validation_report.json"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host "Starting Gate 2 short-train suite: $RunRoot"
Write-Host "baseline subset -> L0-S1 subset"
Write-Host ""

$baselineOut = Join-Path $RunRoot "baseline_subset_shorttrain"
$baselineLog = Join-Path $RunRoot "baseline_subset_shorttrain.log"
$baselineCmd = @(
    $Python, "-u", "-m", "src.train_3dgs",
    "--imgdir", $BaselineImgDir,
    "--colmap", $BaselineSparse,
    "--outdir", $baselineOut,
    "--iterations", "5000",
    "--eval-steps", "1000",
    "--grow-grad2d", "0.0008",
    "--validation-report", $BaselineReport,
    "--disable-video"
)
Write-Host "[1/2] baseline subset short-train"
& $Python $RunAndTee --log $baselineLog -- @baselineCmd

$l0Out = Join-Path $RunRoot "l0_s1_subset_shorttrain"
$l0Log = Join-Path $RunRoot "l0_s1_subset_shorttrain.log"
$l0Cmd = @(
    $Python, "-u", "-m", "src.train_3dgs",
    "--imgdir", $L0ImgDir,
    "--colmap", $L0Sparse,
    "--outdir", $l0Out,
    "--iterations", "5000",
    "--eval-steps", "1000",
    "--grow-grad2d", "0.0008",
    "--validation-report", $L0Report,
    "--disable-video"
)
Write-Host ""
Write-Host "[2/2] L0-S1 subset short-train"
& $Python $RunAndTee --log $l0Log -- @l0Cmd

Write-Host ""
Write-Host "[OK] Gate 2 short-train suite finished: $RunRoot"
