param(
    [string]$SourceScene = "C:\3d-recon-pipeline\outputs\experiments\train_probes\u_base_mcmc_capmax_aa_fulltrain_20260420_032355\mcmc_capmax_750k_aa\_colmap_scene",
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa"
)

$ErrorActionPreference = "Stop"

$imagesSrc = Join-Path $SourceScene "images"
$sparseSrc = Join-Path $SourceScene "sparse\0"

if (-not (Test-Path -LiteralPath $imagesSrc)) {
    throw "Source images path not found: $imagesSrc"
}
if (-not (Test-Path -LiteralPath $sparseSrc)) {
    throw "Source sparse path not found: $sparseSrc"
}

$datasetSceneRoot = Join-Path $SandboxRoot "data\$DatasetName\$SceneName"
$imagesDst = Join-Path $datasetSceneRoot "images"
$sparseDst = Join-Path $datasetSceneRoot "sparse\0"
$outputsRoot = Join-Path $SandboxRoot "outputs"

New-Item -ItemType Directory -Path $imagesDst -Force | Out-Null
New-Item -ItemType Directory -Path $sparseDst -Force | Out-Null
New-Item -ItemType Directory -Path $outputsRoot -Force | Out-Null

Get-ChildItem -LiteralPath $imagesSrc -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $imagesDst -Recurse -Force
}
Get-ChildItem -LiteralPath $sparseSrc -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $sparseDst -Recurse -Force
}

$meta = [ordered]@{
    prepared_at = (Get-Date).ToString("s")
    source_scene = $SourceScene
    sandbox_root = $SandboxRoot
    dataset_name = $DatasetName
    scene_name = $SceneName
    image_count = (Get-ChildItem -LiteralPath $imagesDst -File | Measure-Object).Count
    sparse_files = (Get-ChildItem -LiteralPath $sparseDst -File | Select-Object -ExpandProperty Name)
}

$metaPath = Join-Path $datasetSceneRoot "probe_manifest.json"
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding UTF8

Write-Host "[OK] Scaffold-GS probe data prepared"
Write-Host "Scene root: $datasetSceneRoot"
Write-Host "Manifest:   $metaPath"
