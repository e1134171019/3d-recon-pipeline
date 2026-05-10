param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [string]$Gpu = "0",
    [int]$Iterations = 7000,
    [ValidateSet("cuda", "cpu")]
    [string]$DataDevice = "cpu",
    [double]$VoxelSize = 0.01,
    [int]$Ratio = 5,
    [int]$Resolution = 8,
    [int]$AppearanceDim = 0,
    [double[]]$LambdaDssimValues = @(0.1, 0.2, 0.3),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$launcher = "C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_windows.ps1"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Launcher missing: $launcher"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "[MATRIX] Scaffold-GS lambda_dssim probe batch"
Write-Host "Sandbox:      $SandboxRoot"
Write-Host "Dataset:      $DatasetName"
Write-Host "Scene:        $SceneName"
Write-Host "DataDevice:   $DataDevice"
Write-Host "GPU:          $Gpu"
Write-Host "Iterations:   $Iterations"
Write-Host "Ratio:        $Ratio"
Write-Host "Resolution:   $Resolution"
Write-Host "VoxelSize:    $VoxelSize"
Write-Host "AppearanceD:  $AppearanceDim"
Write-Host "DSSIM sweep:  $($LambdaDssimValues -join ', ')"

foreach ($lambdaValue in $LambdaDssimValues) {
    $tag = ($lambdaValue.ToString("0.0")).Replace(".", "")
    $experimentName = "scaffold_dssim{0}_r{1}_res{2}_v{3}_app{4}_7k_{5}" -f $tag, $Ratio, $Resolution, (($VoxelSize).ToString("0.000").Replace(".", "")), $AppearanceDim, $timestamp
    $argumentList = @(
        "-File", $launcher,
        "-SandboxRoot", $SandboxRoot,
        "-DatasetName", $DatasetName,
        "-SceneName", $SceneName,
        "-ExperimentName", $experimentName,
        "-Gpu", $Gpu,
        "-Iterations", $Iterations,
        "-VoxelSize", $VoxelSize,
        "-Ratio", $Ratio,
        "-Resolution", $Resolution,
        "-AppearanceDim", $AppearanceDim,
        "-LambdaDssim", $lambdaValue,
        "-DataDevice", $DataDevice
    )

    Write-Host ""
    Write-Host ("[lambda_dssim={0}] {1}" -f $lambdaValue, $experimentName)

    if ($DryRun) {
        Write-Host "  dry-run only"
        continue
    }

    & "C:\Program Files\PowerShell\7\pwsh.exe" @argumentList
}
