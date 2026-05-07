param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [string]$ExperimentName = "probe_win_20260507",
    [string]$Gpu = "0"
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

Push-Location $repoRoot
try {
    & $venvPython train.py --eval -s $sceneRoot --lod 0 --gpu $Gpu --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 7000 -m $outputRoot
}
finally {
    Pop-Location
}
