<#
.SYNOPSIS
    MCMC Tier S 過夜批次 orchestrator。

.DESCRIPTION
    讀取 scripts/mcmc_marathon_configs.json，依序跑每個 probe，fail-soft，
    結果即時寫到 results.csv（包含跑出來的 LPIPS / PSNR / SSIM）。
    全程使用既有 src/train_3dgs.py 入口，不複製任何訓練邏輯。

    架構：
      1. 讀 configs.json，依 -Tier 過濾
      2. 每個 probe：
           a. 組 train_3dgs CLI args（沿用現有 wrapper）
           b. 透過 scripts/run_and_tee.py 跑 + log
           c. 從 outputs/.../stats/val_stepXXXX.json 抓 LPIPS
           d. 即時 append 到 results.csv
      3. 失敗繼續下一個（不 throw）
      4. 全部跑完印 summary 表

.PARAMETER Tier
    過濾要跑的層級："S" / "A" / "B" / "" (= 全部)

.PARAMETER Resume
    從第 N 個 probe 開始（1-based），用於停電 / 中斷後續跑

.PARAMETER ConfigPath
    configs.json 路徑，預設 scripts/mcmc_marathon_configs.json

.PARAMETER DryRun
    僅印出每個 probe 的命令，不實際執行（用於驗證 wrapper diff）

.PARAMETER RetryFailed
    自動偵測 OutputRoot 下最新 batch 目錄，從 results.csv 找出
    status=fail（或 exception）的 probe，僅重跑這些 probe 並更新原
    results.csv 對應 row。配合 -RetryBatchDir 可指定特定 batch。

.PARAMETER RetryBatchDir
    搭配 -RetryFailed 使用，明確指定要重跑的 batch 目錄絕對路徑。
    省略時自動取 OutputRoot 下最新的 batch_*。

.PARAMETER OutputRoot
    output 根目錄，預設 outputs/experiments/mcmc_marathon

.EXAMPLE
    # 跑全部 Tier S（6 個，~24 小時）
    .\scripts\run_mcmc_marathon.ps1 -Tier S

.EXAMPLE
    # 從第 3 個 probe 續跑（停電後）
    .\scripts\run_mcmc_marathon.ps1 -Tier S -Resume 3

.EXAMPLE
    # 全部 dry-run，看會送什麼 CLI
    .\scripts\run_mcmc_marathon.ps1 -DryRun

.EXAMPLE
    # 重跑最新一輪 batch 中失敗的 probe（自動找 latest batch_*）
    .\scripts\run_mcmc_marathon.ps1 -Tier S -RetryFailed
#>

param(
    [string]$Tier = "",
    [int]$Resume = 1,
    [string]$ConfigPath = "",
    [switch]$DryRun,
    [switch]$RetryFailed,
    [string]$OutputRoot = "",
    [string]$RetryBatchDir = ""
)

$ErrorActionPreference = "Stop"

# ── 路徑解析 ─────────────────────────────────────
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunAndTee = Join-Path $Root "scripts\run_and_tee.py"

if (-not $ConfigPath) {
    $ConfigPath = Join-Path $Root "scripts\mcmc_marathon_configs.json"
}
if (-not (Test-Path $ConfigPath)) {
    throw "找不到 marathon configs：$ConfigPath"
}
if (-not $DryRun) {
    if (-not (Test-Path $Python)) {
        throw "找不到 Python 環境：$Python（請先建好 .venv）"
    }
    if (-not (Test-Path $RunAndTee)) {
        throw "找不到 run_and_tee.py：$RunAndTee"
    }
}

# ── 讀 config ─────────────────────────────────────
$Config = Get-Content $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Globals = $Config.global_defaults

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $Root $Globals.outdir_root
}

# 過濾 tier
$AllProbes = @($Config.probes)
$Probes = if ($Tier) {
    $AllProbes | Where-Object { $_.tier -eq $Tier }
} else {
    $AllProbes
}

if (-not $Probes -or $Probes.Count -eq 0) {
    throw "Tier='$Tier' 沒有任何 probe。可用 tier：$(($AllProbes | Select-Object -ExpandProperty tier -Unique) -join ', ')"
}

# Resume 跳過
if ($Resume -gt 1) {
    if ($Resume -gt $Probes.Count) {
        throw "Resume=$Resume 超過 probe 總數 $($Probes.Count)"
    }
    $Probes = $Probes[($Resume - 1)..($Probes.Count - 1)]
    Write-Host "[Resume] 從第 $Resume 個 probe 開始"
}

# ── lib_bilagrid pre-check（如有 probe 用 bilateral grid，需先確認可 import）──
function Test-LibBilagridAvailable {
    param([string]$PythonPath)
    if (-not (Test-Path $PythonPath)) {
        return $false
    }
    $checkScript = 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("lib_bilagrid") else 1)'
    & $PythonPath -c $checkScript 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

$NeedsBilagrid = @($Probes | Where-Object { $_.use_bilateral_grid })
$BilagridAvailable = $true
if ($NeedsBilagrid.Count -gt 0 -and -not $DryRun) {
    $BilagridAvailable = Test-LibBilagridAvailable -PythonPath $Python
    if (-not $BilagridAvailable) {
        Write-Host "[bilagrid] lib_bilagrid 在 venv 中找不到。受影響的 probe 將被標記為 skipped（不視為 fail）。" -ForegroundColor Yellow
        Write-Host "[bilagrid] 安裝指令：" -ForegroundColor Yellow
        Write-Host "  Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/examples/lib_bilagrid.py' -OutFile '$Root\gsplat_runner\lib_bilagrid.py'" -ForegroundColor Yellow
    }
}

# ── batch 目錄 ─────────────────────────────────────
# RetryFailed 模式：找最新 batch_* dir，讀 results.csv，過濾出 fail probe，沿用同 BatchDir 重寫該 row。
$ExistingResults = $null
if ($RetryFailed) {
    $TargetBatchDir = $RetryBatchDir
    if (-not $TargetBatchDir) {
        $candidates = Get-ChildItem -Path $OutputRoot -Directory -Filter "batch_*" -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTime -Descending
        if (-not $candidates -or $candidates.Count -eq 0) {
            throw "-RetryFailed 找不到任何 batch_* 目錄於 $OutputRoot"
        }
        $TargetBatchDir = $candidates[0].FullName
    }
    if (-not (Test-Path $TargetBatchDir)) {
        throw "-RetryFailed 指定的 batch 不存在：$TargetBatchDir"
    }
    $BatchDir = $TargetBatchDir
    $LogsDir = Join-Path $BatchDir "logs"
    $ResultsCsv = Join-Path $BatchDir "results.csv"
    if (-not (Test-Path $ResultsCsv)) {
        throw "-RetryFailed 找不到 results.csv：$ResultsCsv"
    }

    $ExistingResults = Import-Csv -Path $ResultsCsv -Encoding UTF8
    $FailedIds = @($ExistingResults | Where-Object { $_.status -in @("fail", "exception") } |
                   Select-Object -ExpandProperty id)
    if ($FailedIds.Count -eq 0) {
        Write-Host "[RetryFailed] $ResultsCsv 中沒有 fail/exception row，無事可做。" -ForegroundColor Yellow
        return
    }
    Write-Host "[RetryFailed] 重跑目錄：$BatchDir"
    Write-Host "[RetryFailed] 失敗 probe IDs：$($FailedIds -join ', ')"
    $Probes = @($Probes | Where-Object { $FailedIds -contains $_.id })
    if ($Probes.Count -eq 0) {
        throw "-RetryFailed 找到 fail rows，但 configs.json 中沒有對應 ID（可能 configs 已改名）。fail IDs: $($FailedIds -join ', ')"
    }
} else {
    $BatchTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BatchDir = Join-Path $OutputRoot "batch_$BatchTimestamp"
    $LogsDir = Join-Path $BatchDir "logs"
    $ResultsCsv = Join-Path $BatchDir "results.csv"
}

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

# ── 預估總時間 ────────────────────────────────────
$TotalEta = ($Probes | Measure-Object -Property expected_eta_hours -Sum).Sum
$EtaFinish = (Get-Date).AddHours($TotalEta)

Write-Host ""
Write-Host "========================================"
Write-Host "  MCMC Marathon Batch"
Write-Host "========================================"
Write-Host "Config       : $ConfigPath"
Write-Host "Tier filter  : $(if ($Tier) { $Tier } else { '(all)' })"
Write-Host "Resume from  : $Resume"
Write-Host "Retry failed : $RetryFailed"
Write-Host "Probe count  : $($Probes.Count)"
Write-Host "Total ETA    : $TotalEta hours"
Write-Host "ETA finish   : $($EtaFinish.ToString('yyyy-MM-dd HH:mm'))"
Write-Host "Batch dir    : $BatchDir"
Write-Host "Results CSV  : $ResultsCsv"
Write-Host "Dry run      : $DryRun"
if ($NeedsBilagrid.Count -gt 0) {
    Write-Host "Bilagrid req : $($NeedsBilagrid.Count) probe，可用=$BilagridAvailable"
}
Write-Host "========================================"
Write-Host ""

# CSV header：RetryFailed 模式不重寫 header，沿用原 CSV；其他情況預先寫好。
if (-not $RetryFailed) {
    "id,tier,iterations,start_time,end_time,duration_hours,status,lpips,psnr,ssim,output_dir,log_file,error" |
        Out-File -FilePath $ResultsCsv -Encoding UTF8
}

# ── 工具函數 ───────────────────────────────────────

function Build-TrainArgs {
    param(
        [Parameter(Mandatory)] $Probe,
        [Parameter(Mandatory)] $Globals,
        [Parameter(Mandatory)] [string]$OutDir
    )

    $imgdirAbs = Join-Path $Root $Globals.imgdir
    $colmapAbs = Join-Path $Root $Globals.colmap
    $reportAbs = Join-Path $Root $Globals.validation_report

    $args = @(
        "--train-mode", $Probe.train_mode,
        "--imgdir", $imgdirAbs,
        "--colmap", $colmapAbs,
        "--outdir", $OutDir,
        "--iterations", "$($Probe.iterations)",
        "--sh-degree", "$($Globals.sh_degree)",
        "--eval-steps", "$($Globals.eval_steps)",
        "--data-factor", "$($Globals.data_factor)",
        "--validation-report", $reportAbs,
        "--cap-max", "$($Probe.cap_max)"
    )

    if ($Probe.antialiased) { $args += "--antialiased" }
    if ($Globals.disable_video) { $args += "--disable-video" }

    # MCMC-specific knobs (從 PR1 暴露的新 CLI flag)
    if ($null -ne $Probe.mcmc_refine_stop_iter) {
        $args += @("--mcmc-refine-stop-iter", "$($Probe.mcmc_refine_stop_iter)")
    }
    if ($null -ne $Probe.ssim_lambda) {
        $args += @("--ssim-lambda", "$($Probe.ssim_lambda)")
    }
    if ($Probe.use_bilateral_grid) { $args += "--use-bilateral-grid" }
    if ($Probe.depth_loss) { $args += "--depth-loss" }
    # Note: when with_ut is true, train_3dgs.py auto-emits --with-eval3d.
    if ($Probe.with_ut) { $args += "--with-ut" }

    if ($Probe.extra_args) {
        foreach ($extra in $Probe.extra_args) { $args += $extra }
    }

    return ,$args
}

function Get-LatestMetrics {
    param([Parameter(Mandatory)] [string]$OutDir)

    # gsplat simple_trainer 把 val metrics 寫到 stats/val_stepXXXX.json
    $statsDir = Join-Path $OutDir "stats"
    if (-not (Test-Path $statsDir)) {
        return @{ lpips = $null; psnr = $null; ssim = $null }
    }
    $latest = Get-ChildItem $statsDir -Filter "val_step*.json" -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        return @{ lpips = $null; psnr = $null; ssim = $null }
    }
    try {
        $j = Get-Content $latest.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        return @{
            lpips = $j.lpips
            psnr = $j.psnr
            ssim = $j.ssim
        }
    } catch {
        return @{ lpips = $null; psnr = $null; ssim = $null }
    }
}

function Wait-GpuRelease {
    param([int]$Seconds = 30)
    Write-Host "[GPU cooldown] sleep $Seconds seconds..."
    Start-Sleep -Seconds $Seconds
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    }
}

# ── 主迴圈 ───────────────────────────────────────
$Results = @()
$Index = $Resume

foreach ($probe in $Probes) {
    $probeId = $probe.id
    $tier = $probe.tier
    $iterations = $probe.iterations
    $eta = $probe.expected_eta_hours

    $start = Get-Date
    Write-Host ""
    Write-Host "----------------------------------------"
    Write-Host "[$($start.ToString('HH:mm'))] [$Index/$($Probes.Count + $Resume - 1)] BEGIN $probeId (tier=$tier, iter=$iterations, eta=${eta}h)"
    Write-Host "  rationale: $($probe.rationale)"
    Write-Host "----------------------------------------"

    $probeOutDir = Join-Path $BatchDir $probeId
    $probeLog = Join-Path $LogsDir "$probeId.log"
    New-Item -ItemType Directory -Force -Path $probeOutDir | Out-Null

    $trainArgs = Build-TrainArgs -Probe $probe -Globals $Globals -OutDir $probeOutDir

    $cmd = @($Python, "-u", "-m", "src.train_3dgs") + $trainArgs

    Write-Host "Command: $($cmd -join ' ')"
    Write-Host "Log:     $probeLog"

    $status = "skipped"
    $errMsg = ""
    $metrics = @{ lpips = $null; psnr = $null; ssim = $null }

    if ($DryRun) {
        Write-Host "[DryRun] 不實際執行"
        $status = "dryrun"
    } elseif ($probe.use_bilateral_grid -and -not $BilagridAvailable) {
        $status = "skipped"
        $errMsg = "lib_bilagrid_unavailable"
        Write-Host "[skip] $probeId 需要 lib_bilagrid，請依上方指令安裝後重跑。" -ForegroundColor Yellow
    } else {
        try {
            & $Python $RunAndTee --log $probeLog -- @cmd
            if ($LASTEXITCODE -eq 0) {
                $status = "success"
                $metrics = Get-LatestMetrics -OutDir $probeOutDir
            } else {
                $status = "fail"
                $errMsg = "exit_code=$LASTEXITCODE"
            }
        } catch {
            $status = "exception"
            $errMsg = $_.Exception.Message
            Write-Host "[ERROR] $probeId 例外：$errMsg" -ForegroundColor Red
        }
    }

    $end = Get-Date
    $duration = [math]::Round(($end - $start).TotalHours, 3)

    $row = [PSCustomObject]@{
        id = $probeId
        tier = $tier
        iterations = $iterations
        start_time = $start.ToString("yyyy-MM-dd HH:mm:ss")
        end_time = $end.ToString("yyyy-MM-dd HH:mm:ss")
        duration_hours = $duration
        status = $status
        lpips = $metrics.lpips
        psnr = $metrics.psnr
        ssim = $metrics.ssim
        output_dir = $probeOutDir
        log_file = $probeLog
        error = $errMsg
    }
    $Results += $row

    # 即時持久化 row：
    # - RetryFailed 模式：把 results.csv 中同 probe id 的舊 fail row 換成新 row（read+rewrite，O(n) 但 csv 通常 <100 row）。
    # - 一般模式：直接 append（停電也保住前面結果）。
    $line = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},"{10}","{11}","{12}"' -f `
        $row.id, $row.tier, $row.iterations, $row.start_time, $row.end_time, `
        $row.duration_hours, $row.status, `
        $(if ($null -ne $row.lpips) { $row.lpips } else { "" }), `
        $(if ($null -ne $row.psnr) { $row.psnr } else { "" }), `
        $(if ($null -ne $row.ssim) { $row.ssim } else { "" }), `
        $row.output_dir, $row.log_file, $row.error
    if ($RetryFailed) {
        $existingLines = Get-Content -Path $ResultsCsv -Encoding UTF8
        $header = $existingLines[0]
        $body = @($existingLines | Select-Object -Skip 1 |
                  Where-Object { -not $_.StartsWith("$($row.id),") })
        $body += $line
        @($header) + $body | Out-File -FilePath $ResultsCsv -Encoding UTF8
    } else {
        Add-Content -Path $ResultsCsv -Value $line -Encoding UTF8
    }

    Write-Host ""
    Write-Host "[$($end.ToString('HH:mm'))] END $probeId — status=$status, duration=${duration}h, lpips=$($row.lpips)"
    Write-Host ""

    $Index++

    # GPU cooldown 介於 probe 之間（最後一個不需要）
    if (-not $DryRun -and $probe -ne $Probes[-1]) {
        Wait-GpuRelease -Seconds 30
    }
}

# ── 最終 summary ────────────────────────────────
Write-Host ""
Write-Host "========================================"
Write-Host "  Marathon Complete"
Write-Host "========================================"
$Results | Format-Table -Property id, tier, status, duration_hours, lpips, psnr, ssim -AutoSize

$success = ($Results | Where-Object { $_.status -eq "success" }).Count
$failed = ($Results | Where-Object { $_.status -in @("fail","exception") }).Count
Write-Host ""
Write-Host "Success: $success / $($Results.Count)"
Write-Host "Failed : $failed / $($Results.Count)"

# 找出最低 LPIPS（成功的 runs 中）
$ranked = $Results | Where-Object { $_.status -eq "success" -and $null -ne $_.lpips } |
          Sort-Object lpips
if ($ranked) {
    $best = $ranked[0]
    Write-Host ""
    Write-Host "=== Best LPIPS ==="
    Write-Host "  id        : $($best.id)"
    Write-Host "  lpips     : $($best.lpips)"
    Write-Host "  psnr      : $($best.psnr)"
    Write-Host "  output_dir: $($best.output_dir)"
}

Write-Host ""
Write-Host "Full results : $ResultsCsv"
Write-Host "Logs         : $LogsDir"
Write-Host ""
