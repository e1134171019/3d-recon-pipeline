$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$sandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe"
$pythonExe = Join-Path $sandboxRoot "scaffold_venv\Scripts\python.exe"
$runner = Join-Path $sandboxRoot "run_viewrobust_train_pr.py"
$vcvarsall = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$cmdExe = "C:\Windows\System32\cmd.exe"
$sdkVersion = "10.0.26100.0"
$sdkIncludeRoot = "C:\Program Files (x86)\Windows Kits\10\Include\$sdkVersion"
$sdkLibRoot = "C:\Program Files (x86)\Windows Kits\10\Lib\$sdkVersion"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Sandbox python missing: $pythonExe"
}
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Runner missing: $runner"
}
if (-not (Test-Path -LiteralPath $vcvarsall)) {
    throw "vcvarsall missing: $vcvarsall"
}
if (-not (Test-Path -LiteralPath $cmdExe)) {
    throw "cmd.exe missing: $cmdExe"
}

Set-Location $sandboxRoot

$envLines = & $cmdExe /c "`"$vcvarsall`" x64 && set" 2>&1
foreach ($ln in $envLines) {
    if ($ln -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}

$sdkIncludeParts = @(
    (Join-Path $sdkIncludeRoot "ucrt"),
    (Join-Path $sdkIncludeRoot "shared"),
    (Join-Path $sdkIncludeRoot "um")
)
$sdkLibParts = @(
    (Join-Path $sdkLibRoot "ucrt\x64"),
    (Join-Path $sdkLibRoot "um\x64")
)

$env:INCLUDE = (($env:INCLUDE -split ';') + $sdkIncludeParts | Where-Object { $_ } | Select-Object -Unique) -join ';'
$env:LIB = (($env:LIB -split ';') + $sdkLibParts | Where-Object { $_ } | Select-Object -Unique) -join ';'

Write-Host "[RUN] View-Robustness PR training probes: T9 + T10"
Write-Host "Sandbox: $sandboxRoot"
Write-Host "Python:  $pythonExe"
Write-Host "Runner:  $runner"
Write-Host "VCVars:  $vcvarsall"
Write-Host "CmdExe:  $cmdExe"
Write-Host "SDK:     $sdkVersion"
Write-Host ""

& $pythonExe $runner --exp T9 T10
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "[DONE] launch_viewrobust_t9_t10.ps1 exit code = $exitCode"
exit $exitCode
