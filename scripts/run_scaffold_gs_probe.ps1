param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [string]$ExperimentName = "probe_20260507",
    [string]$Gpu = "0"
)

$ErrorActionPreference = "Stop"

$repoRoot = Join-Path $SandboxRoot "repo\Scaffold-GS"
$singleTrain = Join-Path $repoRoot "single_train.sh"
$datasetSceneRoot = Join-Path $SandboxRoot "data\$DatasetName\$SceneName"

if (-not (Test-Path -LiteralPath $repoRoot)) {
    throw "Scaffold-GS repo not found: $repoRoot"
}
if (-not (Test-Path -LiteralPath $singleTrain)) {
    throw "single_train.sh not found: $singleTrain"
}
if (-not (Test-Path -LiteralPath $datasetSceneRoot)) {
    throw "Prepared dataset scene not found: $datasetSceneRoot"
}

Write-Host "[NEXT] Scaffold-GS repo is present."
Write-Host "[NEXT] Prepared scene: $datasetSceneRoot"
Write-Host "[MANUAL] Review single_train.sh and map these variables:"
Write-Host "        scene     = $DatasetName/$SceneName/"
Write-Host "        exp_name  = $ExperimentName"
Write-Host "        gpu       = $Gpu"
Write-Host ""
Write-Host "[MANUAL] Training should be launched in an isolated visible terminal after dependency validation."
