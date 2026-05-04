# A 路線第一輪實驗紀錄

**狀態**: current  
**日期**: 2026-04-09  
**目的**: 從「調參找答案」改成「設計實驗找主因」

---

## 實驗設定

本輪固定不變：

- `data/frames_1600`
- baseline SfM sparse model：
  `outputs/runs/baseline_agent_fullchain_20260409_0115/SfM_models/sift/sparse/2`
- baseline SfM 驗證報告：
  `outputs/runs/baseline_agent_fullchain_20260409_0115/reports/pointcloud_validation_report.json`

本輪比較的不是上游 SfM，而是 **3DGS train 端的參數族**。

---

## 四組設計

### 1. `A_base`

目的：

- 作為 train 端 baseline

參數：

- `iterations = 30000`
- `densify_until = 15000`
- `absgrad = False`
- `grow_grad2d = 0.0002`
- `antialiased = False`
- `random_bkgd = False`

### 2. `A_qty`

目的：

- 驗證單純增加訓練量與 densify 截止是否有效

參數：

- `iterations = 40000`
- `densify_until = 20000`
- `absgrad = False`
- `grow_grad2d = 0.0002`
- `antialiased = False`
- `random_bkgd = False`

### 3. `A_absgrad_06`

目的：

- 驗證 split/clone 觸發邏輯是否是主因

參數：

- `iterations = 40000`
- `densify_until = 20000`
- `absgrad = True`
- `grow_grad2d = 0.0006`
- `antialiased = False`
- `random_bkgd = False`

### 4. `A_visual_06`

目的：

- 驗證 `antialiased=True` 是否只是視覺補償，還是會改變訓練指標

參數：

- `iterations = 40000`
- `densify_until = 20000`
- `absgrad = True`
- `grow_grad2d = 0.0006`
- `antialiased = True`
- `random_bkgd = False`

---

## 實驗結果

### 已完成三組

| 組別 | PSNR | SSIM | LPIPS | num_GS | train_time_sec |
|------|------|------|-------|--------|----------------|
| `A_base` | `24.3053` | `0.8601` | `0.2158` | `3,557,023` | `3399.892` |
| `A_qty` | `23.9350` | `0.8345` | `0.2508` | `4,317,912` | `5166.627` |
| `A_absgrad_06` | `20.6456` | `0.7549` | `0.3699` | `3,838,046` | `4804.538` |

來源：

- `outputs/experiments/a_route_matrix/comparison_report.json`

### 第四組處理方式

`A_visual_06` 未納入完整結論，原因如下：

- 前三組已經明確指出方向
- `A_base` 明顯優於 `A_qty` 與 `A_absgrad_06`
- 第四組早期訊號偏弱

截至 `val_step2999.json` 的早期訊號：

- `PSNR = 19.1994`
- `SSIM = 0.7233`
- `LPIPS = 0.4174`
- `num_GS = 944,931`

因此決定中止第四組，避免把 GPU 時間繼續花在低價值分支上。

---

## 解讀

### 1. `A_base` 是本輪相對最佳，但仍未達標

`A_base` 目前只是本輪裡最不差的組，**不代表已達到地圖品質標準**。

關鍵問題仍在：

- `LPIPS` 仍偏高
- Unity 端仍有稀疏與破碎感
- 仍未達到「可交付地圖基底」標準

### 2. 單純加量不是答案

`A_qty` 相對 `A_base`：

- `num_GS` 明顯增加
- 但 `PSNR / SSIM / LPIPS` 反而變差

這表示：

**增加步數與 densify 截止，不等於品質提升。**

### 3. `absgrad=True + grow_grad2d=0.0006` 不是這批資料的直接解法

`A_absgrad_06` 整體指標最差，表示至少在這批資料、這組設定下：

- 質化 densification 沒有帶來改善
- 甚至可能讓分布更不穩

### 4. `num_GS` 不是越多越好

這輪很清楚：

- `A_base` 的 `num_GS` 較少，但品質最好
- `A_qty` 的 `num_GS` 最多，但品質更差

所以 `num_GS` 在這裡應視為：

**模型複雜度 / 分布診斷指標**

不是：

**越多越好的分數**

---

## 本輪結論

本輪實驗已經證明：

1. 問題不是單純步數不夠
2. 本輪 `absgrad_06` 也不是正確方向
3. `antialiasing` 分支在目前訊號下優先級下降

因此下一輪 follow-up 改成：

- 保留 `A_base` 的骨幹
- 測：
  - `grow_grad2d = 0.0008`
  - `random_bkgd = True`

這一輪的核心目的不再是重複驗證 `A_qty` 或 `A_absgrad_06`，而是檢查：

**在不破壞 A_base 穩定性的前提下，是否能透過背景隨機化與更高 grow 門檻改善品質。**

---

## Rerun Summary (2026-05-04 — 2026-05-05)

- Rerun: `A_base_randombg` + `antialiased` — output: `outputs/experiments/a_route_rerun_matrix/A_base_randombg_aa/3DGS_models`
  - `val_step0999`: PSNR 18.808 / SSIM 0.7039 / LPIPS 0.479
  - `val_step1999`: PSNR 19.331 / SSIM 0.7200 / LPIPS 0.434
  - `val_step2999`: PSNR 19.984 / SSIM 0.7429 / LPIPS 0.398
  - `val_step3999`: PSNR 20.2033 / SSIM 0.7511 / LPIPS 0.37897
  - file: [val_step3999.json](outputs/experiments/a_route_rerun_matrix/A_base_randombg_aa/3DGS_models/stats/val_step3999.json#L1)
  - 結論：相較於 `A_base_randombg` 同步驟的指標（PSNR 20.2489 / SSIM 0.7558 / LPIPS 0.3680），本次交叉組合略微落後，但表現持續回升，屬於「值得保留但需更多步驗證」。

- 新啟動（2026-05-05）: `A_base_grow08_randombg` + `antialiased` — output: `outputs/experiments/a_route_rerun_matrix/A_base_grow08_randombg_aa/3DGS_models`
  - 設定重點：`grow_grad2d=0.0008`、`antialiased=True`、`random_bkgd=True`
  - 初始狀態：已啟動訓練並等待首輪評估（1000/2000/3000 步）
  - 路徑：`outputs/experiments/a_route_rerun_matrix/A_base_grow08_randombg_aa/3DGS_models`

下一步：
- 等待 `A_base_grow08_randombg_aa` 的 1000/2000/3000 步評估落地，然後比較同步驟指標再決定是否繼續跑完整 30k。

