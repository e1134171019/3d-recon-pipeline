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
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$launcher = "C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_windows.ps1"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Launcher missing: $launcher"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$matrix = @(
    @{ Name = "scaffold_featbank0_r5_res8_v0010_app0_7k"; UseFeatBank = $false },
    @{ Name = "scaffold_featbank1_r5_res8_v0010_app0_7k"; UseFeatBank = $true }
)

Write-Host "[MATRIX] Scaffold-GS use_feat_bank probe batch"
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

foreach ($entry in $matrix) {
    $experimentName = "{0}_{1}" -f $entry.Name, $timestamp
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
        "-DataDevice", $DataDevice
    )
    if ($entry.UseFeatBank) {
        $argumentList += "-UseFeatBank"
    }

    Write-Host ""
    Write-Host ("[use_feat_bank={0}] {1}" -f $entry.UseFeatBank, $experimentName)

    if ($DryRun) {
        Write-Host "  dry-run only"
        continue
    }

    & "C:\Program Files\PowerShell\7\pwsh.exe" @argumentList
}
