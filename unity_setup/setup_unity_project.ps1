# setup_unity_project.ps1
# 用 Unity batch mode 建立專案、安裝 Gaussian Splatting 插件、匯入 splat 檔
# 使用方式：.\unity_setup\setup_unity_project.ps1

$UNITY_EXE    = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe"
$PROJECT_PATH = "C:\FactoryScene"
$SPLAT_SRC    = "C:\3d-recon-pipeline\exports\3dgs"
$LOG_DIR      = "C:\3d-recon-pipeline\unity_setup\logs"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $PROJECT_PATH | Out-Null

Write-Host "=== Step 1: 建立 Unity 專案 ===" -ForegroundColor Cyan
$createLog = "$LOG_DIR\create_project.log"
$proc = Start-Process -FilePath $UNITY_EXE `
    -ArgumentList "-batchmode", "-nographics", "-createProject", $PROJECT_PATH, "-quit" `
    -RedirectStandardOutput $createLog `
    -RedirectStandardError  "$LOG_DIR\create_project_err.log" `
    -Wait -PassThru
Write-Host "建立專案結果：$($proc.ExitCode)"

# Step 2: 寫入 manifest.json（加入 Gaussian Splatting 插件）
Write-Host "=== Step 2: 安裝 Gaussian Splatting 插件 ===" -ForegroundColor Cyan
$manifestPath = "$PROJECT_PATH\Packages\manifest.json"

$manifest = @{
    dependencies = [ordered]@{
        "com.aras-p.gaussian-splatting" = "https://github.com/aras-p/UnityGaussianSplatting.git?path=package"
        "com.unity.render-pipelines.universal" = "17.0.3"
        "com.unity.modules.jsonserialize" = "1.0.0"
    }
} | ConvertTo-Json -Depth 5
$manifest | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host "manifest.json 已寫入"

# Step 3: 複製 splat 檔案到 Unity Assets
Write-Host "=== Step 3: 複製訓練輸出至 Unity Assets ===" -ForegroundColor Cyan
$destAssets = "$PROJECT_PATH\Assets\GaussianSplats"
New-Item -ItemType Directory -Force -Path $destAssets | Out-Null

$plyFiles = Get-ChildItem $SPLAT_SRC -Filter "*.ply" -Recurse -ErrorAction SilentlyContinue
if ($plyFiles) {
    $plyFiles | Copy-Item -Destination $destAssets -Force
    Write-Host "複製 $($plyFiles.Count) 個 .ply 檔案到 $destAssets"
} else {
    Write-Host "[警告] 尚未找到 .ply 檔案，請確認訓練已完成" -ForegroundColor Yellow
    Get-ChildItem $SPLAT_SRC -Recurse -ErrorAction SilentlyContinue | Select-Object FullName
}

# Step 4: 放入 Editor 匯入腳本
Write-Host "=== Step 4: 安裝 Editor C# 腳本 ===" -ForegroundColor Cyan
$editorDir = "$PROJECT_PATH\Assets\Editor"
New-Item -ItemType Directory -Force -Path $editorDir | Out-Null
Copy-Item "C:\3d-recon-pipeline\unity_setup\GaussianSplatImporter.cs" `
          "$editorDir\GaussianSplatImporter.cs" -Force
Write-Host "Editor 腳本已複製"

# Step 5: 開啟 Unity（正常 GUI 模式）供你查看結果
Write-Host ""
Write-Host "=== 完成！正在開啟 Unity Editor ===" -ForegroundColor Green
Write-Host "專案路徑：$PROJECT_PATH"
Write-Host "Splat 檔案：$destAssets"
Write-Host ""
Start-Process -FilePath $UNITY_EXE -ArgumentList "-projectPath", $PROJECT_PATH
