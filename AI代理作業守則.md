# AI 指引與專案戒律 (AI Instructions)

> 這是一份專為 Agent (AI 助手) 設計的上下文注入文件。在每次對話前，必須優先讀取本份文件。

## 1. 專案主軸與 Agent 的核心職責
- **專案主軸**：這不僅是一個 3DGS 腳本，而是一套包含「影片抽幀 -> Unity 場景匯出」的**端到端全自動化 A/B 對照管線 (Pipeline)**。
- **Agent 本身的角色**：**介入地圖建立與訓練決策！** AI 需看懂 A/B 對照結果，產出報告，並在管線各階段 (L0, SfM, 3DGS) 給予下一步的參數與排障建議。
- **Unity 開發邊界**：
  - ✅ **可做**：透過 PowerShell/batch 自動化腳本，將 `.ply` 批量匯入為 `GaussianSplatAsset` 及建立 Unity 場景（當前 Phase 2）。
  - ❌ **絕對不可做**：人類/機器人行為模擬、物理碰撞、狀態監測等「應用層」邏輯（保留至下階段擴充，目前嚴禁介入）。

### 1B. 生產層 / 決策層正式邊界
- **生產層**：`C:\3d-recon-pipeline`
  - 執行 `Phase 0 -> SfM -> 3DGS -> export / Unity`
  - 產出正式 reports、artifacts、metrics
- **決策層**：`D:\agent_test`
  - 只讀正式 contract / event
  - 只做 Gate 判斷、Recovery 建議、Phase 報告
- **Runtime 正式接口**：
  - `outputs/agent_events/latest_sfm_complete.json`
  - `outputs/agent_events/latest_train_complete.json`
  - `outputs/agent_events/latest_export_complete.json`
  - `outputs/agent_decisions/latest_sfm_decision.json`
  - `outputs/agent_decisions/latest_train_decision.json`
  - `outputs/agent_decisions/latest_export_decision.json`
- 決策層不得再以舊式固定路徑掃描整個 `outputs/` 當主入口；正式依據是 stage contract。
- 生產層若要讀回 Gate 結果，只能讀 `outputs/agent_decisions/latest_*_decision.json`，不得直接耦合 `D:\agent_test\outputs\phase0\...`。
- **保留規則**：
  - 保留：`latest_*.json`、`run_root/reports/agent_<stage>.json`、`D:\agent_test\outputs\phase0\<run_id>\<stage>\*`
  - 可清理：`outputs/agent_events/<timestamp>_*.json`、`outputs/agent_decisions/<timestamp>_*.json`
  - 原則：共享 inbox / outbox 只保留 `latest_*` 當正式狀態；完整歷史由 per-run contract 與決策層審計目錄承擔。
  - 清理腳本：`python scripts/cleanup_agent_mailboxes.py --apply`

## 2. 嚴密的 Gate 0~3 快速驗證協定
為了不浪費算力，Agent 評估任何新增實驗（如 L0 洗幀）時，**必須絕對遵守**以下閥門標準，未過關者直接停止，嚴禁盲目推薦 full training：
- **Gate 0 (Sanity)**：檢查 L0 清洗後的高光比例、保留特徵數、ROI 主體不被截斷。
- **Gate 1 (SfM 幾何)**：只跑 SfM 子集，看 `registered_images`, `points3D`, `inlier ratio` 是否更穩。
- **Gate 2 (早期優勢)**：只跑 `5000 iter` 短訓練，觀察 PSNR/LPIPS 是否有早期改善。
- **Gate 3 (Full Train)**：只有通過上述所有早期優勢關卡，才獲准跑 `30000 iter` 並觀測最終成績。

## 2B. 正式 A/B 對照最小欄位
任何被視為「正式可比較」的 A/B 對照，至少必須包含以下三層欄位。若缺欄位，只能算局部觀察，不能升格為正式結論。

### SfM 層
- `registered_images`
- `points3D`
- camera / image coverage
- geometry stability
- feature / matching runtime

### 3DGS 層
- `PSNR`
- `SSIM`
- `LPIPS`
- `num_GS`
- training runtime

### Unity / 實際使用層
- 主體完整度
- 浮點 / 白點 / 雜點量
- 表面連續度
- 是否可作為後續人機模擬底圖

### 比較戒律
- 不可直接拿不同 validation view set 的 `LPIPS` 做嚴格比較。
- 不可拿被污染的 run（例如 scene dir / frame count 不一致）當正式證據。
- 正式結論優先使用：
  - 同一資料 regime
  - 同一 eval views
  - 同一 checkpoint 或同一 full-train 終點

## 3. 當前技術主線與黑名單 (Current State)
- **訓練主線 (3DGS)**：目前打敗 `U_base` 0.205x 天花板的最佳策略為官方 **`MCMCStrategy`** (`mcmc preset`: LPIPS 0.19187)。接下來的重點是 1M 顆高斯的「Unity 匯出成本評估」或 `cap_max` 限制。
- **黑名單避雷**：`app_opt=True`, `sh_degree=1`, `pose_opt=True`, 與 `machine-level loss mask` 等已被標記為弱訊號或負面因素，Agent 不應主動提出作為優化建議。

## 4. 絕對戒律 (CRITICAL RULES)
- **輸出路徑**：全線強制使用 `outputs/`，絕對禁止寫入或依賴舊版 `exports/`。
- **環境與補丁紅線**：系統環境嚴格鎖定 Python 3.12, CUDA 12.8, gsplat 1.5.3。若遇到 gsplat 的 JIT 編譯報錯，必須先查閱是否為此專案的 **RTX 5070 Ti (sm_120)** 特殊支援問題，並遵守記錄在冊的「4 個標準補丁」原則解決，**嚴禁**自己憑空推測亂改 `TORCH_CUDA_ARCH_LIST` 或是胡亂瞎修 `cpp_extension.py`。

## 4B. Coverage 口徑
- 正式 coverage 只看產品主線：
  - `src/preprocess_phase0.py`
  - `src/downscale_frames.py`
  - `src/sfm_colmap.py`
  - `src/train_3dgs.py`
  - `src/export_ply.py`
  - `src/export_ply_unity.py`
- 不把以下內容混入正式 coverage：
  - `outputs/**`
  - `scripts/**`
  - `experimental/**`
  - `gsplat_runner/**`
  - `src/run_*.py`
  - `scripts/test_cuda.py`
  - `unity_setup/**`
- 正式設定以 `.coveragerc` 為準。

## 5. 全域連動修改守則 (Global Update Rule)
當要求修改核心邏輯、更換框架或推進新的開發階段時，Agent **不准直接開始寫程式**。Agent 必須先執行以下動作：
1. 列出「被影響的程式碼清單」。
2. 列出「哪幾份主線說明書 (如 PROJECT_VISION, SETUP) 的『現在狀態』需要被同步覆寫？」。
3. 判斷「這項變更是否會產生新的歷史？（如是，需新增獨立的 ADR 日誌到 `docs/實驗歷史與決策日誌.md` 封存）」。
**將此變動影響分析 (Impact Analysis) 交由使用者審查同意後，才能同步且同時更新程式碼與文件，嚴防上下文腐敗。**

## 6. 實驗收尾與垃圾回收 (Experiment Closure & Garbage Collection)
AI 在進行 A/B 測試時，常會建立臨時腳本 (如 `test_xxx.py`)。為了防止專案覆蓋率變差，實驗結束後**絕對不允許將臨時腳本棄置於專案中**。AI 必須主動執行「收尾溯源四部曲」：
1. **裁定結果**：實驗是成功（準備推動合併）還是失敗（打入冷宮）？
2. **寫入史冊**：將實驗對應的程式碼精神與成敗原因，濃縮寫入 `docs/實驗歷史與決策日誌.md`。
3. **上下文對齊**：若實驗成功，將邏輯合入主程式後，必須回頭檢查 `PROJECT_VISION` 等主線文件，確保新的實作沒有與舊有的上下文說明產生邏輯矛盾。
4. **銷毀測試檔**：確認所有實驗知識皆已安全轉移至「說明書」與「主程式」後，**無條件刪除該臨時測試檔**，保持專案絕對整潔。
