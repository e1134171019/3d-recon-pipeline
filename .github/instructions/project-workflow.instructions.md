---
description: "Use when working on the 3d-recon-pipeline or agent_test projects. Enforce official docs, boundaries, and terminal rules before making changes."
---
# Project Workflow Rules

- 以目前專案結構與正式主線為準，不要依單張圖片、單次對話內容、舊封存文件或舊路徑推斷架構。
- 每次任務開始前，先讀：
  1. 文件導航.md
  2. 專案願景與當前狀態.md §當前狀態
  3. 依任務類型讀對應正式文件
  4. 必要時再讀 README.md
  未完成上述步驟禁止動手。
- 正式治理、正式文件清單、跨層邊界、正式 agent 介面與 coverage 口徑，一律以 `docs/_governance.md` 為準。
- 不得用單張圖片、單次對話、舊封存文件、舊路徑或未列入正式治理的檔案，覆蓋正式主線。
- 長時間任務必須開可見終端；啟動前清查舊 watcher、重複 watcher、殘留 python/colmap/glomap 進程。
- PowerShell 路徑含空格時，一律使用 Start-Process -FilePath 或 & '完整路徑'，不得裸寫命令字串。
- 修改前先列出保留 / 刪除 / 歸檔建議。

## Codex CLI / VS Code 終端規則

- 若要在 VS Code 內使用 Codex CLI，不得用 Start-Process 另開外部終端。
- 應使用 VS Code integrated terminal 或 .vscode/tasks.json 啟動，並在啟動前固定：
  - UTF-8 / chcp 65001
  - Node.js PATH：C:\Program Files\nodejs;C:\Users\User\AppData\Roaming\npm
  - 工作目錄：C:\3d-recon-pipeline
- 外部可見終端只用於長時間訓練、SfM、Unity batch、ffmpeg 等需要獨立觀察輸出的任務。
- Codex CLI 互動式 TUI 預設在 VS Code 內建終端執行。
