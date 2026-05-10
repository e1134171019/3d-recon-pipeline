param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [string]$Gpu = "0",
    [int]$Iterations = 7000,
    [ValidateSet("cuda", "cpu")]
    [string]$DataDevice = "cpu",
    [int[]]$OnlyIndex = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$launcher = "C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_windows.ps1"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Launcher missing: $launcher"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$matrix = @(
    @{ Index = 1; Ratio = 5; Resolution = 4; VoxelSize = 0.01; Name = "scaffold_r5_res4_v001_7k" },
    @{ Index = 2; Ratio = 5; Resolution = 8; VoxelSize = 0.01; Name = "scaffold_r5_res8_v001_7k" },
    @{ Index = 3; Ratio = 10; Resolution = 4; VoxelSize = 0.01; Name = "scaffold_r10_res4_v001_7k" },
    @{ Index = 4; Ratio = 10; Resolution = 8; VoxelSize = 0.01; Name = "scaffold_r10_res8_v001_7k" },
    @{ Index = 5; Ratio = 5; Resolution = 4; VoxelSize = 0.02; Name = "scaffold_r5_res4_v002_7k" },
    @{ Index = 6; Ratio = 10; Resolution = 4; VoxelSize = 0.02; Name = "scaffold_r10_res4_v002_7k" }
)

if ($OnlyIndex.Count -gt 0) {
    $selected = $matrix | Where-Object { $OnlyIndex -contains $_.Index }
    if ($selected.Count -eq 0) {
        throw "OnlyIndex did not match any matrix entry."
    }
    $matrix = $selected
}

Write-Host "[MATRIX] Scaffold-GS 7000-iter probe batch"
Write-Host "Sandbox:    $SandboxRoot"
Write-Host "Dataset:    $DatasetName"
Write-Host "Scene:      $SceneName"
Write-Host "DataDevice: $DataDevice"
Write-Host "GPU:        $Gpu"
Write-Host "Entries:    $($matrix.Count)"

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
        "-VoxelSize", $entry.VoxelSize,
        "-Ratio", $entry.Ratio,
        "-Resolution", $entry.Resolution,
        "-DataDevice", $DataDevice
    )

    Write-Host ""
    Write-Host ("[{0}/6] {1}" -f $entry.Index, $experimentName)
    Write-Host ("  ratio={0} resolution={1} voxel_size={2}" -f $entry.Ratio, $entry.Resolution, $entry.VoxelSize)

    if ($DryRun) {
        Write-Host "  dry-run only"
        continue
    }

    & "C:\Program Files\PowerShell\7\pwsh.exe" @argumentList
}
