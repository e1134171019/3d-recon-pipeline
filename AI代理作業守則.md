# AI 指引與專案戒律 (AI Instructions)

> 這是一份專為 Agent (AI 助手) 設計的上下文注入文件。在每次對話前，必須優先讀取本份文件。

> **共同治理**：請見 [docs/_governance.md](docs/_governance.md)（治理戒律 + 正式 9 份說明書清單 + 跨層接口）。

## 0. Codex CLI / VS Code 終端規則
- 若要在 VS Code 內使用 Codex CLI，不得用 `Start-Process` 另開外部終端；應使用 VS Code integrated terminal 或 `.vscode/tasks.json` 啟動。
- 啟動 Codex CLI 前必須固定 UTF-8、`chcp 65001`、Node.js PATH（`C:\Program Files\nodejs;C:\Users\User\AppData\Roaming\npm`）與工作目錄 `C:\3d-recon-pipeline`。
- 外部可見終端只用於長時間訓練、SfM、Unity batch、ffmpeg 等需要獨立觀察輸出的任務；Codex CLI 互動式 TUI 預設在 VS Code 內建終端執行。

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
- **Stage contract 最小 schema**：生產層寫出 `latest_*_complete.json` 前必須通過 `src/utils/agent_contracts.py` 的最小驗證；決策層讀取 contract / JSON history 必須走 `D:\agent_test\src\contract_io.py`，不得再在各檔手寫 encoding fallback 或 JSONL fallback。
- **必填核心欄位**：`schema_version / timestamp / run_id / run_root / stage / status`；`artifacts / metrics / params` 必須是 object，缺值時只能正規化為 `{}`，不得混入 list 或自然語言段落作為正式 contract。
- 生產層在 `train_complete` / `export_complete` 寫完 `latest_*` event 後，現在會同步觸發 `D:\agent_test\run_phase0.py --contract ...` 嘗試刷新對應 decision outbox。若 hook 失敗，主流程只記警告，不因決策層異常反向中止。
- 決策層不得再以舊式固定路徑掃描整個 `outputs/` 當主入口；正式依據是 stage contract。
- 生產層若要讀回 Gate 結果，只能讀 `outputs/agent_decisions/latest_*_decision.json`，不得直接耦合 `D:\agent_test\outputs\phase0\...`。
- **保留規則**：
  - 保留：`latest_*.json`、`run_root/reports/agent_<stage>.json`、`D:\agent_test\outputs\phase0\<run_id>\<stage>\*`
  - 可清理：`outputs/agent_events/<timestamp>_*.json`、`outputs/agent_decisions/<timestamp>_*.json`
  - 原則：共享 inbox / outbox 只保留 `latest_*` 當正式狀態；完整歷史由 per-run contract 與決策層審計目錄承擔。
  - 清理腳本：`python scripts/cleanup_agent_mailboxes.py --apply`

### 1C. Agent 正式定位與演進原則
- **Agent 系統的本質是輔助生產層，不是專案主體。** 專案真正持續擴大的對象是生產層能力（地圖建立、Unity 驗證、未來的人行為 / 機器狀態 / DXF 工件狀態），不是 agent 機制本身。
- **決策層不是自治多 agent 執行器。** 它的正式角色是：根據正式 `state / event / 實驗史` 產生候選、收斂策略、回寫審計與決策結果，協助生產層減少錯誤實驗與重工。
- **生產層必須可獨立運作。** 即使決策層 hook、validator 或 report 流程失敗，主流程也只能降級為警告與人工判讀，不能反向成為生產層的單點阻塞。
- **Agent 核心只處理抽象決策流程。** 核心只應承擔 `state / event / candidate / decision / feedback / audit`；不得把 `LPIPS`、`cap_max`、Unity 視覺細節等 map-building 專屬 heuristics 寫死成全域核心規則。
- **Phase-specific 細節應以 strategy packs 擴充。** 地圖建立、人行為、機器狀態、DXF 等能力未來都應各自擁有策略模組；核心維持穩定，策略可替換與增量成長。
- **單一 arbiter 負責正式裁決。** 各 strategy module 只能提出候選，不得直接改主線；只有 arbiter 能依正式文件、當前 phase、黑名單與實驗史輸出唯一正式 next step。
- **外部對話框 AI 的定位是 meta evaluator。** 它不取代 runtime 主控，而是用來審查：agent 是否真的持續優化、是否在收斂、是否又被 phase 細節污染核心。

### 1D. Agent V1 改寫計畫
- **先定核心 contract，再改程式。** 在任何 agent 重寫前，先固定 5 個抽象物件：`current_state`、`event`、`candidate`、`arbiter_decision`、`outcome_feedback`。沒有統一 schema，不得直接擴寫多 agent。
- **核心只保留決策流程，不承載地圖建立 heuristics。** 核心只負責讀 state / event、收集 candidates、交給 arbiter、寫 audit / feedback；`LPIPS`、`cap_max`、Unity 霧化等細節只應存在於 strategy pack。
- **現有地圖建立邏輯視為第一個 strategy pack。** 當前 `sfm/train/export/unity/recovery` 的判斷規則，後續應收斂為 `map_building` pack，而不是繼續直接擴寫進 agent 核心。
- **arbiter 只輸出唯一正式 next step。** 多個 strategy module 可以並行提出候選，但正式 outbox 只能有單一 arbiter 裁決，不得讓多個 module 直接覆寫 `latest_*_decision.json`。
- **對話框 AI 只做外部優化審查。** 每輪改寫或策略更新後，應以正式 state / event / decision / feedback 為材料，由外部對話框 AI 檢查系統是否更收斂，而不是把對話框 AI 接成 runtime 主控。
- **V1 遷移順序固定。**
  1. 定 schema
  2. 切核心 / map_building pack 邊界
  3. 抽出 candidates
  4. 補 arbiter contract
  5. 補 outcome feedback
  6. 最後才調整實際 decision logic

### 1E. CoverageStrategy 與 CI Hard Gate
- **Coverage 問題分成兩層處理：agent 分析 + CI 阻擋。** `CoverageStrategy` 是決策層中的 strategy module，只負責讀 coverage 報告、diff 與正式主線範圍，提出 coverage 風險候選；真正阻止低覆蓋率變更進主線的是 CI hard gate，不是 agent 自行裁決。
- **CoverageStrategy 的輸入只看正式主線六模組。** 不得把 `outputs`、`scripts`、`gsplat_runner`、`unity_setup` 混入正式 coverage 分析。
- **CoverageStrategy 的輸出是 finding / candidate，不是直接改碼。** 它應指出：哪個正式主線檔案新增了未覆蓋分支、return path、exception path，建議補哪類測試；不得自行放寬門檻、修改主線或覆寫正式 decision。
- **CoverageStrategy 最小版已落地。** 決策層目前已有 `D:\agent_test\agents\quality\coverage_strategy.py`，可讀 coverage.py JSON 並輸出 findings/candidates；它仍不是 CI gate，也不得自動改生產層程式。
- **CI hard gate 負責把品質閥門制度化。** 正式主線應以 branch coverage、diff coverage 或等效檢查作為 required status check；未達門檻的變更不得進主線。
- **arbiter 只收斂 coverage 候選，不取代 CI。** arbiter 可以依 `CoverageStrategy` 建議決定「先補測試再繼續」或「暫停升格主線」，但最終 merge 保護仍由 CI status checks 承擔。
- **外部對話框 AI 的 coverage 角色仍是 meta evaluator。** 它用來審查：coverage 規則是否真的讓正式主線更穩，是否只是追數字灌水，是否有把不該納入的檔案混進正式 coverage。

### 1F. Agent V1 五個核心 Schema 草案
- **Schema 設計原則：核心欄位 generic、phase 細節外掛。** 核心 schema 只定義決策流程一定需要的欄位；像 `LPIPS`、`cap_max`、Unity 霧化、DXF 幾何約束等 domain 細節，必須放在對應 strategy pack 的 `metrics / evidence / params` 內，不得回滲成全域核心必填欄位。
- **`current_state`**：描述決策當下唯一正式狀態。最小欄位包含：
  - `state_id`
  - `phase`
  - `active_pack`
  - `current_best`
  - `next_focus`
  - `allowed_actions`
  - `blocked_actions`
  - `blacklist`
  - `source_docs`
  - `updated_at`
- **`event`**：描述生產層剛完成的事實。最小欄位包含：
  - `event_id`
  - `run_id`
  - `stage`
  - `pack`
  - `status`
  - `run_root`
  - `artifacts`
  - `metrics`
  - `evidence`
  - `timestamp`
  - `source_contract`
- **`candidate`**：描述單一 strategy module 提出的候選策略。最小欄位包含：
  - `candidate_id`
  - `source_module`
  - `scope`
  - `proposal_type`
  - `title`
  - `rationale`
  - `params`
  - `expected_gain`
  - `expected_risk`
  - `estimated_cost`
  - `blocked_by`
  - `evidence`
  - `confidence`
- **`arbiter_decision`**：描述單一 arbiter 對候選池的正式裁決。最小欄位包含：
  - `decision_id`
  - `state_ref`
  - `event_ref`
  - `selected_candidate_id`
  - `rejected_candidate_ids`
  - `decision`
  - `reason`
  - `next_action`
  - `can_proceed`
  - `requires_human_review`
  - `written_at`
- **`outcome_feedback`**：描述已執行決策的回饋結果，供後續收斂。最小欄位包含：
  - `feedback_id`
  - `decision_ref`
  - `run_id`
  - `outcome_status`
  - `observed_metrics`
  - `observed_artifacts`
  - `drift_vs_expectation`
  - `lessons`
  - `update_targets`
  - `recorded_at`
- **正式約束**：
  - 所有 schema 都必須可序列化成穩定 JSON，不依賴自然語言段落當唯一真相。
  - `latest_*_decision.json` 只允許寫入 `arbiter_decision` 類輸出，不得混入候選池或原始觀察。
  - `outcome_feedback` 不得直接覆寫 `current_state`；狀態更新必須經過正式文件與 arbiter / meta evaluator 審查。
  - 決策層核心輸出必須經 `D:\agent_test\src\contract_io.py` 驗證後才可寫檔；適用範圍包含 `candidate_pool.json`、`current_state.json`、`arbiter_decision.json`、`outcome_feedback.json` 與 `latest_*_decision.json`。
  - `candidate_pool.json` 必須維持 `candidate_count == len(candidates)`；`arbiter_decision.json` 必須包含 `written_at` 與單一正式裁決；production-facing `latest_*_decision.json` 不得混入候選池或自然語言長報告。

### 1G. `agent_test` 現況對照：核心 / map_building pack 切分
- **目前可直接視為核心層的檔案**
  - `D:\agent_test\run_phase0.py`
    - 只負責 entrypoint、CLI 參數、呼叫 runner。
  - `D:\agent_test\src\phase0_runner.py`
    - 只負責 latest contract 掃描、watch mode、啟動 coordinator。
  - `D:\agent_test\src\coordinator.py` 的下列責任
    - contract 讀取
    - shared decision 寫回
    - audit / report 路徑整理
    - event bus
  - `D:\agent_test\src\candidate_pool.py`
    - 把 stage proposal log 統一整理成 `candidate_pool.json`。
  - `D:\agent_test\src\arbiter.py`
    - 根據 `current_state.json + candidate_pool.json` 產出單一 `arbiter_decision.json`。
  - `D:\agent_test\src\current_state.py`
    - 把 contract stage、report 建議與 candidate 摘要整理成 `current_state.json`。
  - `D:\agent_test\src\shared_decision_mapper.py`
    - 把 `arbiter_decision + report state` 映射成生產層相容的 `latest_*_decision.json` payload。
  - `D:\agent_test\src\outcome_feedback.py`
    - 把 arbiter decision、current state、report 與 artifacts 整理成 `outcome_feedback.json`。
- **目前應下沉為 `map_building` strategy pack 的檔案**
  - `agents/phase0/pointcloud_validator.py`
  - `agents/phase0/map_validator.py`
  - `agents/phase0/production_param_gate.py`
  - `agents/phase0/recovery_advisor.py` 與 `agents/phase0/phase_reporter.py` 已於 2026-05-01 經 ablation 驗證後移除；停用後 train/export 的正式 decision、next_action 與 selected candidate 不變。
  - `agents/phase0/unity_param_gate.py` 與 `agents/phase0/unity_importer.py` 已於 2026-05-01 第二輪 ablation 驗證後移除；停用後 latest sfm/train/export 的正式 decision / next_action / selected candidate / dominant layer 無必要改善，且生產層未讀取 decision-layer 的 `unity_export_params.json` 或 `import_summary.json` 作正式接口。
  - 第二輪 ablation 報告：`C:\tmp\agent_stage_ablation\stage_ablation_report.json`。
  - 已修正：coordinator 會在 pack 未寫出 `phase0_report.json` 時，依 `pack_result`、`validation_report` 與 `decision_log` 生成正式 `phase0_report.json`，再交給 `current_state / arbiter / shared decision` 使用。
  - 驗證：2026-05-05 replay latest train/export 後，train = `hold_export` + selected `PPG-001` + dominant `parameter`；export = `hold_phase_close` + selected `PPG-001` + dominant `parameter`。
  - 已整理：`ArtifactResolver` 統一 contract artifact alias / fallback；`Phase0ReportGenerator` 統一 `phase0_report` 生成規則。兩者仍留在 `coordinator.py` 內，不新增核心檔，避免 6-core 膨脹。
  - 已整理：`ProblemLayerAnalyzer` 統一 `problem_layer` 單筆推斷與 candidate aggregation；`candidate_pool.py` 負責單一規則來源，`current_state.py` 只消費聚合結果。
- **現況與 V1 schema 的對照**
  - `event`
    - 已存在：由生產層 `latest_*_complete.json` 與 contract 提供。
    - 問題：pack 名稱與證據欄位尚未統一。
  - `current_state`
    - 已初步一級化：`src/current_state.py` 會把 contract stage、report 建議與 candidate 摘要寫成 `current_state.json`。
    - 仍有缺口：目前仍是 phase0 / map-building 專用摘要，尚未成為 pack-agnostic 的正式全域 state。
  - `candidate`
    - 已初步抽出：`src/candidate_pool.py` 會把各 stage `proposal` 正規化並寫成 `candidate_pool.json`。
    - 仍有缺口：欄位已統一為最小 schema，但 pack 間 `params / evidence / cost` 還沒有更細的 domain 分層。
  - `arbiter_decision`
    - 已初步抽出：`src/arbiter.py` 會根據 `current_state.json + candidate_pool.json` 產出單一 `arbiter_decision.json`。
    - 仍有缺口：目前 arbiter 仍使用 phase0 stage 規則，不是完全 generic 的 pack-agnostic 裁決器。
  - `outcome_feedback`
    - 已結構化：`src/outcome_feedback.py` 會把決策結果、觀察指標、artifacts、drift、lessons 與 preliminary outcome label 寫成 `outcome_feedback.json`。
    - 已形成最小學習閉環：每輪 decision 會同步產生 `learning_curve.json`，並由 `candidate_pool.py` 讀取歷史 `effectiveness_rate / accepted_rate` 影響下一輪 `rank_score`。
- **當前最明顯的混層點**
  - `coordinator.py` 已不再承擔 phase-specific stages，且 shared decision payload 已抽到 `shared_decision_mapper.py`；目前仍保留 shared outbox 寫檔責任，但已更接近純 orchestration kernel。
  - `map_validator.py` 帶有 map-building 專屬指標與自適應閾值，不能升成全域核心。
- **V1 切分目標**
  - 核心保留：entrypoint、contract intake、candidate pool 收集、單一 arbiter、decision outbox、feedback 寫回。
  - `map_building` pack 保留：pointcloud / train quality / param gates / unity export。
  - support-only report / recovery stage 不再預設保留；若不改變正式 decision，就應刪除或合併到現有核心輸出。
  - `current_state.py` 與 `outcome_feedback.py` 已補上最小可用 state / feedback / learning curve 物件；下一步是補人工 outcome label 入口，讓學習曲線能判斷 agent 何時可從 meta-evaluator 降為 observer。

### 1H. Agent 學習曲線與對話框 AI 退出條件
- **學習曲線不是模型訓練曲線，而是決策品質曲線。** 目前 `learning_curve.json` 追蹤每輪 decision 的候選來源、problem layer、是否需人工審查、是否重複錯誤、是否浪費 run，以及估計 token 成本。
- **未標籤的 hold decision 不計入 recommendation success。** `can_proceed=false` 的決策預設 `decision_useful=null`，必須由後續人工或實驗結果補標，避免把「暫停」誤當成功或失敗。
- **candidate ranking 只使用已累積的正式 feedback。** 若某 source module 已有 outcome history，`candidate_pool.py` 會以 `effectiveness_rate` 優先，其次才以 `accepted_rate` 影響 `rank_score`，且 `repeat_error_rate` 會作為負向懲罰。
- **arbiter 的 hold-path 現在正式吃 `rank_score`。** 當同一 `problem_layer` 內有多個候選時，`arbiter.py` 會優先選 `rank_score` 較高者；pass path 仍保留 stage gate 的穩定語義，避免一次把整個裁決器改成純分數機制。
- **ProductionParamGate 已改成真 gate。** 只有 `sfm_plan` 或 `train_plan` 真的產出可執行 rerun 參數時才會 `approved=true`；若只是維持現況或缺資料，正式狀態必須是 `hold_manual_review`，不得再用永遠通過的 `approved=true` 假裝有 gate。
- **ProductionParamGate 現在有正式 gate status。** `evaluate()` 會回傳 `gate_status / reason / sfm_profile / train_profile`；pack 端會依結果分流成 `production_params_ready`、`production_params_hold`、`production_params_failed`。只有 `ready` 才代表可直接回寫生產層做 rerun。
- **AdaptiveThreshold 已接到正式 feedback。** `D:\agent_test\adapters\adaptive_threshold.py` 現在優先讀 `outputs/phase0/*/*/outcome_feedback.json` 重建品質門檻歷史，只有缺少正式 feedback 時才回退舊 `phase0_decisions.log`；之後不得再把舊 log 視為唯一學習來源。
- **PyTorch 模型目前只允許離線實驗。** `D:\agent_test\adapters\pytorch_decision_model.py` 可用正式 `outcome_feedback` 訓練 `decision_useful` 分類器，但在樣本量足夠前，不得直接接管 `arbiter` 或覆寫 `latest_*_decision.json`。
- **對話框 AI 的退出不是關閉，而是降級。** 當最近窗口滿足：足夠決策數、recommendation success rate ≥ 0.70、human override rate < 0.20、repeat error rate < 0.10、critical bad release = 0，才建議由 meta-evaluator 降為 observer-only。
- **目前狀態仍是 keep_meta_evaluator。** 2026-05-01 replay latest train/export 後，`learning_curve.json` 可正常產生，但兩筆皆為 `held_for_review` 且尚未人工標籤，因此仍不能宣稱 agent 已可自主優化。
- **人工標籤必須走 CLI，不手改 JSON。** 若要標記某輪決策是否有用，使用 `D:\agent_test\run_phase0.py --label-feedback <outcome_feedback.json> --decision-useful true|false ...`。CLI 會更新 `outcome_feedback.json` 並重算同一 stage 目錄下的 `learning_curve.json`。

### 1I. Offline Learning 正式分層規則
- **formal runtime 與 offline learning 必須分層。** 正式 `candidate_pool / current_state / arbiter / outcome_feedback` 先保持穩定；`Ollama/Qwen teacher`、歷史回填、`PyTorch` baseline 只能存在於 offline learning 層，不得直接改寫 `latest_*_decision.json`。
- **本機 Ollama 的正式角色是 teacher，不是 runtime arbiter。** 它只負責：
  - 對歷史 run / sandbox probe 產生語意標註
  - 補 `role / issue_type / unity_result / next_recommendation / confidence`
  - 作為對話框 AI 的低 token 本機前置判讀器
- **PyTorch 的正式角色是 offline trainer，不是 test 本體。** 真正學習應在 `adapters/train_teacher_augmented_baseline.py` 或後續 trainer 腳本進行；`tests/test_pytorch_decision_model.py` 只驗證 schema、feature merge、mocked teacher output 與模型介面，不承擔真實 teacher loop 或長時間訓練。
- **Qwen / Ollama 不得直接驅動 formal agent runtime。** 正式 runtime 只吃 contract / event / feedback；Qwen teacher 只能先把資料寫進 `outputs/offline_learning/*`，再由 offline learner 吸收。
- **對話框 AI 的正式角色是 meta evaluator。** 它負責審查：
  - teacher prompt / schema 是否合理
  - offline learner 是否有 target leakage / overfitting
  - 哪些 Scaffold / 新框架 probe 值得繼續
  - 何時可把 offline learner 升成 advisory layer

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
- wrapper 預設值不等於 trainer 最終生效值；遇到 `mcmc` 等 preset 模式時，必須先確認 effective config / preset resolution，再下根因判斷。
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
  - `gsplat_runner/**`
  - `scripts/test_cuda.py`
  - `unity_setup/**`
- 正式設定以 `.coveragerc` 為準。

## 4C. 六主程式模組化規格
- **模組化範圍：六個正式主程式一起規劃，逐支落地。** 六主程式仍是 `src/` 的正式邊界，不因重構新增一批零散入口檔。
- **共同骨架固定為：** `Config -> Paths -> Validate -> BuildCommand/Plan -> Run -> CollectMetrics -> WriteContract`。
- **`main()` 只做 orchestration。** `main()` 應讀取 CLI 參數、呼叫 helper、處理頂層錯誤與輸出最終狀態；不得長期承擔參數合併、路徑推導、命令組裝、metrics 解析與 contract 寫出細節。
- **同型 wrapper 規則：**
  - `preprocess_phase0.py`：Config / Paths / frame plan / filter metrics / report。
  - `downscale_frames.py`：Config / Paths / resize plan / execution summary。
  - `sfm_colmap.py`：Config / Paths / feature-match-map command / validation report / `sfm_complete` contract。
  - `train_3dgs.py`：Config / Paths / trainer args / train metrics / `train_complete` contract。
  - `export_ply.py`：Config / Paths / checkpoint resolve / export command / artifact summary。
  - `export_ply_unity.py`：Config / Paths / reconstruction / Unity transform / `export_complete` contract。
- **第一階段只做檔案內模組化。** 在六主程式穩定前，不新增 `src/train_config.py`、`src/sfm_config.py` 之類新主線檔；若未來真的需要共用 helper，必須先確認不會破壞六主程式 coverage 口徑。
- **重構驗證門檻：** 每改一支主程式都必須跑 `pytest` 與 `.coveragerc` coverage；CLI 行為、agent contract schema、`latest_*` event / decision interface 不得變更。

### 2026-04-27 生產層 coverage baseline
- 驗證對象：`C:\3d-recon-pipeline` 產品主線六模組，不包含 `D:\agent_test`。
- 驗證命令：使用唯一 coverage DB，例如 `$env:COVERAGE_FILE='outputs\logs\.coverage_product_YYYYMMDD_HHMM'; python -m coverage run --rcfile=.coveragerc -m pytest -q tests --tb=short -p no:cacheprovider`，再執行 `coverage report -m` 與 `coverage json`。不要重用被鎖住的 `.coverage_product_current`。
- 驗證結果：`80 passed` + `2 subtests passed`，總覆蓋率 `92%`（1135 covered / 1230 statements）。
- 報告位置：`outputs/logs/coverage_product_report.json`。
- 模組覆蓋率：`export_ply.py` 99%、`export_ply_unity.py` 99%、`train_3dgs.py` 98%、`downscale_frames.py` 91%、`sfm_colmap.py` 88%、`preprocess_phase0.py` 81%。
- 本輪改進：補 `sfm_colmap.py` 的 agent params、main 成功路徑、Step 1/2/3 gate 失敗、Mapper 無輸出、COLMAP subprocess failure、COLMAP/GLOMAP resolver、mapper branch 與 sparse stats fallback 測試，使 `sfm_colmap.py` 由 44% 提升至 88%；補 `train_3dgs.py` 的 params-json、MCMC/default command、scale/mask、stats metrics、decision hook、dependency/input failure 與 subprocess failure 測試，使 `train_3dgs.py` 由 68% 提升至 98%；補 `export_ply_unity.py` 的 CLI/main、params-json、denormalize、Unity 座標轉換、filter 與 decision hook 測試，使 `export_ply_unity.py` 由 41% 提升至 99%；補 `preprocess_phase0.py` 的 video path、CLAHE、blur reject、video-open fail 與 fake video 抽幀/過濾統計，使 `preprocess_phase0.py` 由 32% 提升至 81%；補 `export_ply.py` 的 missing checkpoint、gsplat export 成功與 ImportError fallback 測試，使 `export_ply.py` 由 65% 提升至 99%。
- 下一步優先順序：生產層正式 coverage 已越過 90% 目標；後續只針對實際風險補 `preprocess_phase0.py` 的 CLI 收尾區或 `sfm_colmap.py` 的少數診斷分支，不為追數字新增低價值測試。
- 注意：`D:\agent_test` 需要獨立 coverage 設定與 fixture，不得用本 baseline 代表決策層覆蓋率。

### 2026-04-26 決策層 coverage checkpoint
- 驗證對象：`D:\agent_test` 決策層主線，不包含 `archive/`、`outputs/`、`docs/`、`tests/`。
- 驗證結果：`30 passed`，總覆蓋率 `94%`。
- 報告位置：`D:\agent_test\outputs\coverage\coverage_agent_report.json`。
- 已覆蓋重點：`coordinator.py`、`map_building_pack.py`、`phase0_runner.py`、`contract_io.py`、`coverage_strategy.py`、核心 reducers 與主要 phase0 / quality strategy module 的 return / exception / contract fixture 路徑。
- CoverageStrategy 實測：讀取 `outputs/logs/coverage_product_report.json` 後輸出 `D:\agent_test\outputs\coverage\coverage_strategy_report.json`，目前結果為 `pass`，共 `0` 個 findings、`0` 個 candidates。
- 後續規則：決策層新增 strategy module 時，必須同步補測試，維持 `90%+`，避免為了修 runtime 錯誤而產生未覆蓋的新分支。

### 2026-04-26 跨層文件同步規則
- `D:\agent_test` 只記錄決策層自己的真相：runtime contract、core / strategy pack 邊界、coverage checkpoint、執行 SOP 與審計路徑。
- `C:\3d-recon-pipeline` 的正式 9 份文件只記錄生產層主線、全域治理、實驗歷史、L0/Gate 協定、備用方案與故障排查。
- 若修改決策層 runtime 行為，至少同步更新 `D:\agent_test\.instructions.md`；若同時影響生產層 hook、共享 inbox/outbox 或正式 coverage 口徑，必須同步更新本文件。
- 若只是外部圖片或對話提出的重構建議，先列入候選與風險分析，不得直接升格為正式架構。
- 若重構涉及 `preprocess_phase0.py`、`sfm_colmap.py`、`src/utils/agent_contracts.py` 或 decision contract schema，必須先補測試與 fixture，再考慮改主線。

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

---

## 7. Observer Mode 過渡規則（未啟用）

> ⚠️ **本節目前未啟用。** 啟用條件：`D:\agent_test\outputs\phase0` 的學習曲線達到 `ready_for_ai_observer_mode = true`（見 §0E 門檻）。在此之前，維持 `keep_meta_evaluator` 完整模式。

### 7A. 角色分工（啟用後生效）

| 對話框 AI 保留的工作 | 路由給 Ollama 的工作 |
|---|---|
| 審查 teacher prompt 是否符合 schema | 執行 teacher 標注（`label_historical_backfill_with_ollama.py`）|
| 審查 learner 報告是否有 target leakage | 執行離線訓練（`train_teacher_augmented_baseline.py`）|
| 確認 arbiter decision 是否需要人工覆蓋 | 讀取 contract / 產生候選建議 |
| 判斷是否需要調整實驗方向 | 執行 `build_scaffold_probe_backfill.py` |

對話框 AI 必須親自介入（不可路由 Ollama）的情況：
- `critical_bad_release = true`
- `human_override` 被觸發
- Ollama teacher 的 `run_useful` 與正式 `decision_useful` 方向矛盾
- `bridge_pass` 狀態改變（Bridge Gate 1/2/3）
- 正式 9 份說明書需要變更

### 7B. 強制 4 層回報格式（每次涉及 teacher / learner / agent 必須分層回報）

```
1. formal runtime
   有無新的正式 event（outputs/agent_events/latest_*.json）
   有無新的正式 decision（outputs/agent_decisions/latest_*.json）
   arbiter 裁決：decision / next_action / selected_candidate_id / dominant_layer

2. teacher（Ollama）
   是否已執行標注：Y/N
   輸出哪個 JSONL / 標注筆數 / 錯誤筆數

3. learner（PyTorch offline）
   是否已重訓：Y/N
   輸出哪份 report / dataset_size / LOO accuracy / loss_last

4. meta review（對話框 AI）
   teacher 輸出是否有 target leakage 或語意不一致
   learner LOO 是否虛高（dataset_size < 3 × feature_dim 時必須標記警告）
   是否需要調整實驗方向
   下一步建議（具體、可執行）
```

若 `latest_teacher_loop_status.json` 不存在、未更新、或時間早於最新實驗，不得假設 teacher/learner 已完成，必須明確標記「teacher 狀態未知」。

### 7C. Token 節省規則（啟用後生效）

- 每輪任務只貼摘要：JSON 只貼前 20 行，不貼完整 LOO 清單
- traceback 只貼最後 5 行 + exception 訊息
- 指標更新只貼 delta（「LPIPS 0.189 → 0.187」），不重貼全部指標
- 確認類問題先讓 Ollama 回答，對話框 AI 只做最終 sanity check

### 7D. Observer Mode 退出條件（自動升回 keep_meta_evaluator）

以下任一情況發生，立即退出 observer mode：
- `critical_bad_release = true`
- 最近 5 筆中 `human_override` ≥ 3 筆
- 任何 Bridge Gate 從 pass 變 fail
- 正式 9 份說明書需要治理升級
