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
    [string]$BaseUrl = "http://127.0.0.1:11434"
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

& $PythonExe $buildScript --manifest $Manifest --outputs-root $OutputsRoot --output $SeedOutput
& $PythonExe $labelScript --input $SeedOutput --output $TeacherOutput --model $Model --base-url $BaseUrl

if (-not (Test-Path -LiteralPath $HistoricalTeacher)) {
    Write-Host "[WARN] Historical teacher dataset not found: $HistoricalTeacher"
    Write-Host "[OK] Scaffold seed + teacher outputs were generated, but baseline retrain was skipped."
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

Write-Host "[OK] Scaffold teacher loop completed"
Write-Host "Seed:    $SeedOutput"
Write-Host "Teacher: $TeacherOutput"
Write-Host "Merged:  $MergedOutput"
Write-Host "Report:  $ReportOutput"
