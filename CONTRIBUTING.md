# Contributing to 3d-recon-pipeline

> 此檔案聚集 PR 規則、**全域連動修改守則**（Impact Analysis SOP）與**實驗收尾四部曲**。改任何核心邏輯前先讀本檔案。

> **共同治理**：請見 [docs/_governance.md](docs/_governance.md)（治理戒律 + 正式 9 份說明書清單 + 跨層接口）。

## 1. 三條開工前的紅線

**不准直接開始寫程式**。改核心邏輯 / 換框架 / 推進新開發階段前必須完成：

1. **列出「被影響的程式碼清單」**：列出所有將動到的 .py / config / fixture
2. **列出「需要同步覆寫『現在狀態』的主線說明書」**：例如 `專案願景與當前狀態.md` / `docs/L0洗幀管線設計.md` / `docs/未來路線圖與備用方案.md`
3. **判斷「這項變更是否會產生新的歷史？」**：若是 → 必須新增獨立 ADR 日誌到 [`docs/實驗歷史與決策日誌.md`](docs/實驗歷史與決策日誌.md)

**將此變動影響分析（Impact Analysis）交由使用者審查同意後，才能同步且同時更新程式碼與文件，嚴防上下文腐敗。**

> 完整原文見 [`AI代理作業守則.md § 5 全域連動修改守則`](AI代理作業守則.md)。

## 2. PR 規則

### 2.1 一個 PR 只做一件事

- 不混 SfM 改動 + Trainer 改動
- 不混 spec 改動 + dated 紀錄
- 不混 P0（路徑硬寫等架構修正）+ P1（拆檔重構）+ P2（雜訊清理）

### 2.2 必改清單（依改動類型）

| 改動類型 | 必改檔案 |
|---|---|
| **生產層 6 主程式**（`preprocess_phase0.py` / `sfm_colmap.py` / `train_3dgs.py` / `export_ply_unity.py` / 等）改動 | 主程式 + `docs/實驗歷史與決策日誌.md`（dated ADR）+ `docs/L0洗幀管線設計.md`（若 L0 行為變）+ 對應測試 |
| **跨層接口**（`src/utils/agent_contracts.py` / decision contract schema）改動 | 主程式 + `docs/_governance.md` § 3 跨層接口 + `agent_3D/docs/DECISION_DATASET_SCHEMA.md`（鏡像）|
| **Coverage 口徑** | `tools/coverage/run_coverage.py` + `AI代理作業守則.md § 4B Coverage 口徑` + `.coveragerc` |
| **新框架評估**（如 Scaffold-GS、MCMC variant 等）| `experimental/<framework>_probe/` + `docs/未來路線圖與備用方案.md` § 框架評估 + 若決定升正式 → `docs/實驗歷史與決策日誌.md` 新 ADR |
| **故障排查 / 補丁** | `docs/故障排查與急診室.md` + 對應補丁腳本（`scripts/`）|
| **5070Ti / 環境補丁** | `scripts/apply_5070ti_patches.py`（若 in scope）+ `docs/故障排查與急診室.md` |

### 2.3 commit message 格式

```
<type>: <短摘要>

<為什麼>
<改了什麼>
<驗證方式（含 ablation / pytest 結果）>
```

`<type>` ∈ `{feat, fix, docs, refactor, test, chore}`。

### 2.4 黑名單（**禁止**重新引入）

完整黑名單見 [`AI代理作業守則.md § 3 當前技術主線與黑名單`](AI代理作業守則.md)，常見禁止項：

- `pose_opt`（會崩 LPIPS）
- `app_opt`（會在 Unity 端產生色塊偏移）
- `sh_degree=1`（高光遺失）
- `loss mask`（會導致幾何流失）

引入這些項目的 PR 會被 reject，除非**先**送 ablation 測試報告 + 對 LPIPS / Unity 視覺的影響評估。

## 3. 實驗收尾四部曲（Experiment Closure & Garbage Collection）

A/B 測試會建立臨時腳本（`test_xxx.py`）。實驗結束後**絕對不允許棄置於專案中**。必須執行：

1. **裁定結果**：實驗是成功（推合併）還是失敗（打冷宮）？
2. **寫入史冊**：將實驗對應的程式碼精神與成敗原因，濃縮寫入 [`docs/實驗歷史與決策日誌.md`](docs/實驗歷史與決策日誌.md)
3. **上下文對齊**：若成功，回頭檢查 `專案願景與當前狀態.md` 等主線文件，確保新實作沒與舊有上下文產生邏輯矛盾
4. **銷毀測試檔**：確認所有實驗知識已安全轉移至「說明書」與「主程式」後，**無條件刪除**該臨時測試檔

> 完整原文見 [`AI代理作業守則.md § 6 實驗收尾與垃圾回收`](AI代理作業守則.md)。

## 4. Gate 0~3 驗證協定

任何主線改動的 PR 必須說明過了哪幾個 Gate：

| Gate | 內容 | 通過條件 |
|---|---|---|
| **Gate 0** | Sanity（單元測試 + lint + coverage 不降）| `pytest` 全綠、`tools/coverage/run_coverage.py` 不下降 |
| **Gate 1** | Geometry（COLMAP 重建幾何健康）| 至少一份 small dataset 產 valid `cameras.bin / images.bin / points3D.bin` |
| **Gate 2** | Early Short-train（500-1000 iter，PSNR 上升）| short-train log 顯示 PSNR 單調上升、SSIM 不退 |
| **Gate 3** | Full Training（產 LPIPS / SSIM / PSNR + Unity 匯入無色塊）| LPIPS 不超 baseline 5%、Unity 視覺無明顯瑕疵 |

> 完整 Gate 協定 + Mermaid 流程見 [`docs/L0洗幀管線設計.md`](docs/L0洗幀管線設計.md)。

## 5. 提交前 checklist

- [ ] 已完成 §1 三條紅線（影響分析交給使用者審查）
- [ ] 一個 PR 只做一件事（§2.1）
- [ ] 必改檔案清單已涵蓋（§2.2）
- [ ] 沒引入黑名單項（§2.4）
- [ ] 若有臨時測試檔，已執行四部曲（§3）
- [ ] 已說明過了哪幾個 Gate（§4）
- [ ] PR 描述含「為什麼 / 改什麼 / 驗證方式」三段
