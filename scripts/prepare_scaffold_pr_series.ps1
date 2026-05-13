param(
    [string]$ManifestPath = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe\scaffold_pr_series.json",
    [string]$OutputDir = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe\pr_drafts",
    [string]$PrId = "",
    [switch]$CreateBranch
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\3d-recon-pipeline"
$gitExe = "C:\Program Files\Git\cmd\git.exe"

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$prs = @($manifest.pull_requests)
if ($PrId) {
    $prs = @($prs | Where-Object { $_.pr_id -eq $PrId })
    if ($prs.Count -eq 0) {
        throw "Unknown pr_id: $PrId"
    }
}

function New-MarkdownList {
    param([object[]]$Items)
    if (-not $Items -or $Items.Count -eq 0) {
        return "- none"
    }
    return ($Items | ForEach-Object { "- $_" }) -join "`r`n"
}

function New-QuotedPathList {
    param([object[]]$Items)
    if (-not $Items -or $Items.Count -eq 0) {
        return @()
    }
    return @($Items | ForEach-Object { "'$_'" })
}

foreach ($pr in ($prs | Sort-Object order)) {
    $bodyPath = Join-Path $OutputDir ($pr.pr_id + ".md")
    $metaPath = Join-Path $OutputDir ($pr.pr_id + ".json")
    $addListPath = Join-Path $OutputDir ($pr.pr_id + ".git-add.txt")
    $commitMsgPath = Join-Path $OutputDir ($pr.pr_id + ".commit-message.txt")
    $stageScriptPath = Join-Path $OutputDir ($pr.pr_id + ".stage.ps1")

    $summaryBlock = New-MarkdownList -Items @($pr.summary)
    $filesBlock = New-MarkdownList -Items @($pr.files)
    $verifyBlock = New-MarkdownList -Items @($pr.verify_commands)
    $dependsBlock = New-MarkdownList -Items @($pr.depends_on)
    $quotedFiles = New-QuotedPathList -Items @($pr.files)
    $filesLine = $quotedFiles -join " "
    $commitSubject = if ($pr.PSObject.Properties.Name -contains "commit_subject") { $pr.commit_subject } else { $pr.title }

    $body = @"
# $($pr.title)

## Branch
`$($pr.branch_name)`

## PR ID
`$($pr.pr_id)`

## Scope
$summaryBlock

## Files
$filesBlock

## Verification
$verifyBlock

## Depends On
$dependsBlock

## Notes
- Series: `$($manifest.series_name)`
- Scope: `$($manifest.scope)`
- Prepared by script: `scripts/prepare_scaffold_pr_series.ps1`
- Default mode only prepares the draft; no automatic commit or push.
"@

    $body | Set-Content -LiteralPath $bodyPath -Encoding utf8
    (@($pr.files) -join "`r`n") | Set-Content -LiteralPath $addListPath -Encoding utf8

    $commitBody = @"
$commitSubject

$summaryBlock

Files:
$filesBlock

Verification:
$verifyBlock
"@
    $commitBody | Set-Content -LiteralPath $commitMsgPath -Encoding utf8

    $stageScript = @"
param(
    [switch]`$CreateBranch
)

`$ErrorActionPreference = 'Stop'
`$repoRoot = 'C:\3d-recon-pipeline'
`$gitExe = 'C:\Program Files\Git\cmd\git.exe'

if (`$CreateBranch) {
    & `$gitExe -C `$repoRoot checkout -b '$($pr.branch_name)'
}

& `$gitExe -C `$repoRoot add -- $filesLine
& `$gitExe -C `$repoRoot commit -F '$commitMsgPath'
"@
    $stageScript | Set-Content -LiteralPath $stageScriptPath -Encoding utf8

    $meta = [ordered]@{
        series_name = $manifest.series_name
        pr_id = $pr.pr_id
        order = $pr.order
        branch_name = $pr.branch_name
        title = $pr.title
        commit_subject = $commitSubject
        files = @($pr.files)
        verify_commands = @($pr.verify_commands)
        depends_on = @($pr.depends_on)
        body_path = $bodyPath
        add_list_path = $addListPath
        commit_message_path = $commitMsgPath
        stage_script_path = $stageScriptPath
    }
    $meta | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metaPath -Encoding utf8

    if ($CreateBranch) {
        if (-not (Test-Path -LiteralPath $gitExe)) {
            throw "git.exe not found: $gitExe"
        }
        & $gitExe -C $repoRoot checkout -b $pr.branch_name
    }

    Write-Host "[PR-DRAFT] $($pr.pr_id)"
    Write-Host "  Title : $($pr.title)"
    Write-Host "  Branch: $($pr.branch_name)"
    Write-Host "  Body  : $bodyPath"
    Write-Host "  Meta  : $metaPath"
    Write-Host "  Add   : $addListPath"
    Write-Host "  Commit: $commitMsgPath"
    Write-Host "  Stage : $stageScriptPath"
}
