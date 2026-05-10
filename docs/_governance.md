# 共同治理摘要 (Shared Governance)

> 狀態：Current
> 角色：所有正式說明書（9 份）共享的治理規則與正式來源清單之**單一來源（single source of truth）**。
> 規則：本文件被改動時，視為治理規則升級；所有正式說明書頂端應引用本文件，不得再各自抄寫一份。

---

## 1. 治理戒律（每個 agent / 開發者開工前必讀）

- 以目前專案結構與正式主線為準，不依單張圖片、單次對話、舊封存文件或舊路徑推斷架構。
- 任務開始前先依 [文件導航.md](文件導航.md) 路由，再讀 [專案願景與當前狀態.md](專案願景與當前狀態.md) 的「當前狀態」與任務對應正式文件；需要主線總覽時再讀 [README.md](README.md)。
- 正式來源只有下列 9 份文件（見第 2 節）；舊中文文件與 `docs/experiments/` 不再作為正式決策依據。
- 生產層：`C:\3d-recon-pipeline`；決策層：`D:\agent_test`；正式接口只看 `outputs/agent_events/latest_*_complete.json` 與 `outputs/agent_decisions/latest_*_decision.json`。
- 長任務必須開可見終端；PowerShell 空格路徑只用 `Start-Process -FilePath` 或 `& '完整路徑'`；coverage 只看正式主線六模組；修改前先列保留 / 刪除 / 歸檔建議。

---

## 2. 正式來源 — 9 份說明書清單

> 「9 份」是固定數字。本表是**唯一**的正式文件清單。

| # | 路徑 | 角色 |
|---|---|---|
| 1 | [`文件導航.md`](文件導航.md) | 路由表（目的 → 應讀文件）|
| 2 | [`README.md`](README.md) | 主線總覽 + quickstart |
| 3 | [`專案願景與當前狀態.md`](專案願景與當前狀態.md) | 戰略願景 + 當前 state |
| 4 | [`AI代理作業守則.md`](AI代理作業守則.md) | Agent 行為憲法 + V1 schema |
| 5 | [`docs/L0洗幀管線設計.md`](L0洗幀管線設計.md) | L0 設計 + Phase 0 + Gate 0~3 |
| 6 | [`docs/實驗歷史與決策日誌.md`](實驗歷史與決策日誌.md) | ADR / 實驗紀錄 |
| 7 | [`docs/安裝與環境建置.md`](安裝與環境建置.md) | 環境安裝 SOP |
| 8 | [`docs/故障排查與急診室.md`](故障排查與急診室.md) | 排障 + Unity SOP + 5070Ti 補丁 |
| 9 | [`docs/未來路線圖與備用方案.md`](未來路線圖與備用方案.md) | 備用方案 + 框架評估 |

> 連結為相對路徑：本檔位於 `docs/`，因此 1~4 為 `..` 級別；本表使用無 `..` 顯示僅為閱讀方便。

實際相對路徑：

| # | 從 `docs/_governance.md` 看 |
|---|---|
| 1 | `../文件導航.md` |
| 2 | `../README.md` |
| 3 | `../專案願景與當前狀態.md` |
| 4 | `../AI代理作業守則.md` |
| 5 | `./L0洗幀管線設計.md` |
| 6 | `./實驗歷史與決策日誌.md` |
| 7 | `./安裝與環境建置.md` |
| 8 | `./故障排查與急診室.md` |
| 9 | `./未來路線圖與備用方案.md` |

> **不在此表的檔案不是正式說明書**，包含：
> - `地圖優先開發說明書.md`（已封存）
> - `專案願景草稿_封存.md`（已封存）
> - `docs/experiments/*.md`（5 份，僅遷移來源）
> - `experimental/scaffold_gs_probe/README.md`（sandbox sub-doc）
> - 其他根目錄與 `docs/` 內未列入本表的所有檔案

---

## 3. 跨層接口（生產層 / 決策層）

> 此節定義兩倉通訊規則，與「9 份說明書」之治理一致。

| 方向 | 路徑 | 用途 |
|---|---|---|
| 生產 → 決策 | `outputs/agent_events/latest_sfm_complete.json` | SfM 階段事件 |
| 生產 → 決策 | `outputs/agent_events/latest_train_complete.json` | Train 階段事件 |
| 生產 → 決策 | `outputs/agent_events/latest_export_complete.json` | Export 階段事件 |
| 決策 → 生產 | `outputs/agent_decisions/latest_sfm_decision.json` | SfM 決策回寫 |
| 決策 → 生產 | `outputs/agent_decisions/latest_train_decision.json` | Train 決策回寫 |
| 決策 → 生產 | `outputs/agent_decisions/latest_export_decision.json` | Export 決策回寫 |

`latest_*` 檔案為共享 inbox / outbox；per-run 歷史（`<run_id>/<stage>/...`）位於決策層 audit root，**不屬於正式 inbox**。

---

## 4. 變更規則

- 本檔案的變更 = 治理規則升級。
- 任一變更必須同步更新 `AI代理作業守則.md§5「全域連動修改守則」` 描述。
- 不允許其他文件再內聯整段「共同治理摘要」；應直接引用本檔案：
  - 範例（生產層根目錄文件）：`> **共同治理**：請見 [docs/_governance.md](docs/_governance.md)`
  - 範例（`docs/` 內文件）：`> **共同治理**：請見 [_governance.md](_governance.md)`

---

## 5. 為什麼有這份檔案

歷史問題：原先「共同治理摘要」5 行 bullet 被機械複製到 8 份文件開頭。任何一條規則升級都需要同步改 8 個位置；漏改任何一個就是 drift。

決策（2026-05-10）：抽出本檔案作單一來源；其他文件首段改為一行引用。
