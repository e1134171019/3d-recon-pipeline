$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "outputs\experiments\l0_mask_loss\h_d0_maskloss_compare_$Timestamp"

$ImgDir = Join-Path $Root "outputs\experiments\l0_selection_compare\l0_2x2_20260418_020711\H_D0\selected_frames"
$Sparse = Join-Path $Root "outputs\experiments\l0_selection_compare\l0_2x2_20260418_020711\H_D0\SfM_models\l0_s1_windowed\sparse\1"
$Report = Join-Path $Root "outputs\experiments\l0_selection_compare\l0_2x2_20260418_020711\H_D0\reports\pointcloud_validation_report.json"
$MaskDir = Join-Path $RunRoot "loss_masks"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host "Starting H_D0 mask-loss compare: $RunRoot"
Write-Host "Step 1: generate machine-level loss masks"
Write-Host "Step 2: H_D0 unmasked short-train (5000 iter)"
Write-Host "Step 3: H_D0 masked short-train (5000 iter)"
Write-Host ""

$maskLog = Join-Path $RunRoot "generate_masks.log"
$maskCmd = @(
    $Python, "-u", "-m", "src.generate_machine_loss_masks",
    "--imgdir", $ImgDir,
    "--outdir", $MaskDir
)
Write-Host "[1/3] Generate machine loss masks"
& $Python $RunAndTee --log $maskLog -- @maskCmd

$plainOut = Join-Path $RunRoot "h_d0_shorttrain_plain"
$plainLog = Join-Path $RunRoot "h_d0_shorttrain_plain.log"
$plainCmd = @(
    $Python, "-u", "-m", "src.train_3dgs",
    "--imgdir", $ImgDir,
    "--colmap", $Sparse,
    "--outdir", $plainOut,
    "--iterations", "5000",
    "--eval-steps", "1000",
    "--grow-grad2d", "0.0008",
    "--validation-report", $Report,
    "--disable-video"
)
Write-Host ""
Write-Host "[2/3] H_D0 short-train without loss mask"
& $Python $RunAndTee --log $plainLog -- @plainCmd

$maskedOut = Join-Path $RunRoot "h_d0_shorttrain_masked"
$maskedLog = Join-Path $RunRoot "h_d0_shorttrain_masked.log"
$maskedCmd = @(
    $Python, "-u", "-m", "src.train_3dgs",
    "--imgdir", $ImgDir,
    "--colmap", $Sparse,
    "--outdir", $maskedOut,
    "--iterations", "5000",
    "--eval-steps", "1000",
    "--grow-grad2d", "0.0008",
    "--validation-report", $Report,
    "--disable-video",
    "--loss-mask-dir", $MaskDir
)
Write-Host ""
Write-Host "[3/3] H_D0 short-train with machine-level loss mask"
& $Python $RunAndTee --log $maskedLog -- @maskedCmd

Write-Host ""
Write-Host "[OK] H_D0 mask-loss compare finished: $RunRoot"
