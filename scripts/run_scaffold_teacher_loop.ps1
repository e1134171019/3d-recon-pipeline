param(
    [string]$PythonExe = "C:\3d-recon-pipeline\.venv\Scripts\python.exe",
    [string]$AgentRoot = "D:\agent_test",
    [string]$Manifest = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe\data\factorygaussian\u_base_750k_aa\probe_manifest.json",
    [string]$OutputsRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe\repo\Scaffold-GS\outputs",
    [string]$SeedOutput = "D:\agent_test\outputs\offline_learning\scaffold_probe_backfill_seed.jsonl",
    [string]$TeacherOutput = "D:\agent_test\outputs\offline_learning\scaffold_probe_backfill_teacher.jsonl",
    [string]$HistoricalTeacher = "D:\agent_test\outputs\offline_learning\historical_run_backfill_teacher.jsonl",
    [string]$MergedOutput = "D:\agent_test\outputs\offline_learning\historical_plus_scaffold_teacher.jsonl",
    [string]$ReportOutput = "D:\agent_test\outputs\offline_learning\historical_plus_scaffold_report.json",
    [string]$Model = "qwen2.5:14b",
    [string]$BaseUrl = "http://127.0.0.1:11434",
    [string]$WatchRunRoot = "",
    [int]$PollSeconds = 30,
    [int]$TimeoutMinutes = 720,
    [string]$SummaryOutput = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe\latest_teacher_loop_status.json"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python not found: $PythonExe"
}

$buildScript = Join-Path $AgentRoot "adapters\build_scaffold_probe_backfill.py"
$labelScript = Join-Path $AgentRoot "adapters\label_historical_backfill_with_ollama.py"
$trainScript = Join-Path $AgentRoot "adapters\train_teacher_augmented_baseline.py"

foreach ($path in @($buildScript, $labelScript, $trainScript)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required agent script not found: $path"
    }
}

function Get-LatestPointCloudPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunRoot
    )

    $pointCloudRoot = Join-Path $RunRoot "point_cloud"
    if (-not (Test-Path -LiteralPath $pointCloudRoot)) {
        return $null
    }

    $candidates = Get-ChildItem -LiteralPath $pointCloudRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'iteration_*' } |
        ForEach-Object {
            $ply = Join-Path $_.FullName "point_cloud.ply"
            if (Test-Path -LiteralPath $ply) {
                [PSCustomObject]@{
                    Iteration = [int](($_.Name -replace '^iteration_', ''))
                    Path = $ply
                }
            }
        } |
        Sort-Object Iteration -Descending

    if ($candidates -and $candidates.Count -gt 0) {
        return $candidates[0].Path
    }
    return $null
}

function Wait-ForScaffoldArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunRoot,
        [Parameter(Mandatory = $true)]
        [int]$PollSeconds,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutMinutes
    )

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    $resultsPath = Join-Path $RunRoot "results.json"
    $outputsLogPath = Join-Path $RunRoot "outputs.log"

    Write-Host "[WATCH] Waiting for Scaffold artifacts..."
    Write-Host "        RunRoot : $RunRoot"
    Write-Host "        Timeout : $TimeoutMinutes min"

    while ((Get-Date) -lt $deadline) {
        $pointCloudPath = Get-LatestPointCloudPath -RunRoot $RunRoot
        if ((Test-Path -LiteralPath $resultsPath) -or $pointCloudPath) {
            return [PSCustomObject]@{
                status = "ready"
                run_root = $RunRoot
                results_path = if (Test-Path -LiteralPath $resultsPath) { $resultsPath } else { "" }
                point_cloud_path = if ($pointCloudPath) { $pointCloudPath } else { "" }
                outputs_log_path = if (Test-Path -LiteralPath $outputsLogPath) { $outputsLogPath } else { "" }
            }
        }

        Start-Sleep -Seconds $PollSeconds
    }

    return [PSCustomObject]@{
        status = "timeout"
        run_root = $RunRoot
        results_path = ""
        point_cloud_path = ""
        outputs_log_path = if (Test-Path -LiteralPath $outputsLogPath) { $outputsLogPath } else { "" }
    }
}

if ($WatchRunRoot) {
    $watchResult = Wait-ForScaffoldArtifacts -RunRoot $WatchRunRoot -PollSeconds $PollSeconds -TimeoutMinutes $TimeoutMinutes
    if ($watchResult.status -ne "ready") {
        $timeoutPayload = [ordered]@{
            status = "watch_timeout"
            triggered = $false
            watch_run_root = $WatchRunRoot
            poll_seconds = $PollSeconds
            timeout_minutes = $TimeoutMinutes
            observed = $watchResult
            generated_at = (Get-Date).ToString("o")
        }
        $summaryDir = Split-Path -Parent $SummaryOutput
        if (-not (Test-Path -LiteralPath $summaryDir)) {
            New-Item -ItemType Directory -Path $summaryDir -Force | Out-Null
        }
        $timeoutPayload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryOutput -Encoding UTF8
        throw "Timed out waiting for Scaffold artifacts in $WatchRunRoot"
    }
}

& $PythonExe $buildScript --manifest $Manifest --outputs-root $OutputsRoot --output $SeedOutput
& $PythonExe $labelScript --input $SeedOutput --output $TeacherOutput --model $Model --base-url $BaseUrl

if (-not (Test-Path -LiteralPath $HistoricalTeacher)) {
    Write-Host "[WARN] Historical teacher dataset not found: $HistoricalTeacher"
    Write-Host "[OK] Scaffold seed + teacher outputs were generated, but baseline retrain was skipped."
    $seedOnlyPayload = [ordered]@{
        status = "seed_and_teacher_only"
        triggered = [bool]$WatchRunRoot
        watch_run_root = $WatchRunRoot
        seed_output = $SeedOutput
        teacher_output = $TeacherOutput
        merged_output = ""
        report_output = ""
        generated_at = (Get-Date).ToString("o")
    }
    $summaryDir = Split-Path -Parent $SummaryOutput
    if (-not (Test-Path -LiteralPath $summaryDir)) {
        New-Item -ItemType Directory -Path $summaryDir -Force | Out-Null
    }
    $seedOnlyPayload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryOutput -Encoding UTF8
    return
}

$mergedDir = Split-Path -Parent $MergedOutput
if (-not (Test-Path -LiteralPath $mergedDir)) {
    New-Item -ItemType Directory -Path $mergedDir -Force | Out-Null
}

$historicalLines = Get-Content -LiteralPath $HistoricalTeacher -Encoding UTF8
$scaffoldLines = Get-Content -LiteralPath $TeacherOutput -Encoding UTF8
@($historicalLines + $scaffoldLines) | Set-Content -LiteralPath $MergedOutput -Encoding UTF8

& $PythonExe $trainScript --input $MergedOutput --output $ReportOutput

$seedCount = @(Get-Content -LiteralPath $SeedOutput -Encoding UTF8 | Where-Object { $_.Trim() }).Count
$teacherCount = @(Get-Content -LiteralPath $TeacherOutput -Encoding UTF8 | Where-Object { $_.Trim() }).Count
$mergedCount = @(Get-Content -LiteralPath $MergedOutput -Encoding UTF8 | Where-Object { $_.Trim() }).Count
$reportSummary = if (Test-Path -LiteralPath $ReportOutput) {
    Get-Content -LiteralPath $ReportOutput -Encoding UTF8 | ConvertFrom-Json
} else {
    $null
}

$summaryPayload = [ordered]@{
    status = "completed"
    triggered = [bool]$WatchRunRoot
    watch_run_root = $WatchRunRoot
    seed_output = $SeedOutput
    teacher_output = $TeacherOutput
    merged_output = $MergedOutput
    report_output = $ReportOutput
    seed_count = $seedCount
    teacher_count = $teacherCount
    merged_count = $mergedCount
    learner_report = if ($reportSummary) { $reportSummary } else { $null }
    generated_at = (Get-Date).ToString("o")
}
$summaryDir = Split-Path -Parent $SummaryOutput
if (-not (Test-Path -LiteralPath $summaryDir)) {
    New-Item -ItemType Directory -Path $summaryDir -Force | Out-Null
}
$summaryPayload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryOutput -Encoding UTF8

Write-Host "[OK] Scaffold teacher loop completed"
Write-Host "Seed:    $SeedOutput"
Write-Host "Teacher: $TeacherOutput"
Write-Host "Merged:  $MergedOutput"
Write-Host "Report:  $ReportOutput"
Write-Host "Summary: $SummaryOutput"
