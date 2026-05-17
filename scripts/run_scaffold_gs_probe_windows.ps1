param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [string]$ExperimentName = "probe_win_20260507",
    [string]$Gpu = "0",
    [int]$Port = 6009,
    [int]$Iterations = 7000,
    [double]$VoxelSize = 0.001,
    [int]$UpdateInitFactor = 16,
    [int]$AppearanceDim = 0,
    [int]$Ratio = 1,
    [int]$Resolution = -1,
    [double]$LambdaDssim = 0.2,
    [double]$MinOpacity = 0.005,
    [switch]$UseFeatBank,
    [ValidateSet("cuda", "cpu")]
    [string]$DataDevice = "cuda"
)

$ErrorActionPreference = "Stop"

$repoRoot = Join-Path $SandboxRoot "repo\Scaffold-GS"
$venvPython = Join-Path $SandboxRoot ".venv_scaffold\Scripts\python.exe"
$sceneRoot = Join-Path $SandboxRoot "data\$DatasetName\$SceneName"
$outputRoot = Join-Path $repoRoot "outputs\$DatasetName\$SceneName\$ExperimentName"

if (-not (Test-Path -LiteralPath $repoRoot)) {
    throw "Repo missing: $repoRoot"
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Sandbox venv missing: $venvPython"
}
if (-not (Test-Path -LiteralPath $sceneRoot)) {
    throw "Scene data missing: $sceneRoot"
}

$env:CUDA_VISIBLE_DEVICES = $Gpu

Write-Host "[RUN] Scaffold-GS probe (Windows sandbox)"
Write-Host "Repo:   $repoRoot"
Write-Host "Scene:  $sceneRoot"
Write-Host "Output: $outputRoot"
Write-Host "Iters:  $Iterations"
Write-Host "Voxel:  $VoxelSize"
Write-Host "Ratio:  $Ratio"
Write-Host "Res:    $Resolution"
Write-Host "DSSIM:  $LambdaDssim"
Write-Host "MinOp:  $MinOpacity"
Write-Host "FeatBk: $($UseFeatBank.IsPresent)"
Write-Host "Data:   $DataDevice"
Write-Host "Port:   $Port"

Push-Location $repoRoot
try {
    $trainArgs = @(
        "train.py",
        "--eval",
        "-s", $sceneRoot,
        "--lod", "0",
        "--gpu", $Gpu,
        "--voxel_size", $VoxelSize,
        "--update_init_factor", $UpdateInitFactor,
        "--appearance_dim", $AppearanceDim,
        "--ratio", $Ratio,
        "--resolution", $Resolution,
        "--lambda_dssim", $LambdaDssim,
        "--min_opacity", $MinOpacity,
        "--data_device", $DataDevice,
        "--port", $Port,
        "--iterations", $Iterations,
        "-m", $outputRoot
    )
    if ($UseFeatBank) {
        $trainArgs += "--use_feat_bank"
    }

    & $venvPython @trainArgs
}
finally {
    Pop-Location
}
