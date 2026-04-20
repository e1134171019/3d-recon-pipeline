# Phase-0 真實場景掃描器 — 技術說明書

> ⚠️ **封存文件（Archived）**
>
> 本文件已不再是正式來源。其仍有價值的內容已整理移植到：
> - `docs/L0洗幀管線設計.md`
> - `docs/故障排查與急診室.md`
> - `docs/未來路線圖與備用方案.md`
>
> 本文件現在只保留作為歷史實錄與遷移來源，不應再作為當前主線規格或 SOP 依據。

> 版本：v1.4  
> 更新：2026-03-05（§十一 gsplat Windows 編譯全流程完成 ✅；Phase-0A 訓練驗證成功）  
> 對應研究：中小企業板金製造現場智慧化系統（ICG 框架）

---

## 一、這個 Pipeline 在研究中的定位

### 問題背景

Phase-0 的核心挑戰是 **B06（冷啟動零標注資料）**：系統上線前沒有任何現場標注資料，但訓練模型需要大量有標注的圖像。傳統解法是「先手工在 Unity 建模，再生成合成資料」，但這個方法有兩個致命問題：

1. **幾何不準確**：Unity 場景靠手工量尺寸建出，和真實折床機有 sim-to-real gap（T12）
2. **相機位置靠猜**：不知道最佳攝影機安裝角度，T03 無法在部署前驗證

**本 pipeline 同時解決這兩個問題：**

> 用手機/相機繞著折床機拍一段影片  
> → COLMAP SfM 自動算出精確幾何 + 真實相機姿態  
> → 直接匯入 Unity → Unity 場景幾何來自真實世界

---

## 二、整體架構

```
影片（.mp4）
    │
    ▼ [1] extract_frames.py         優先 ffmpeg，備援 OpenCV
data/frames/                        ← 12fps 抽幀
    │
    ▼ [2] downscale_frames.py       長邊縮到 ≤ 1000px
data/frames_1k/                     ← COLMAP 效能優化版
    │
    ▼ [3] sfm_colmap.py             COLMAP SfM
data/work/sparse/0/                 ← 稀疏點雲 + 相機姿態
    │  ├── cameras.bin              相機內參（焦距/畸變）
    │  ├── images.bin               每幀的相機外參（位置+方向）
    │  └── points3D.bin             場景稀疏點雲
    │
    ▼ [4] export_unity_pytorch.py
exports/
    ├── scene.ply                   ← 匯入 Unity 的點雲
    └── cameras.txt                 ← 3DGS / NeRF 訓練用相機格式
```

---

## 三、各模組說明

### [1] extract_frames.py — 影片抽幀

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `fps` | 每秒抽幾幀 | 12 |
| 優先工具 | ffmpeg（速度快、品質穩） | — |
| 備援工具 | OpenCV（無 ffmpeg 時自動切換） | — |

**拍攝建議：**
- 繞機台走一圈，速度平穩（不要快速移動）
- 保持機台各角落都出現在影片中
- 光線均勻（避免強逆光）
- 建議時長：60-120 秒，12fps 約 720-1440 幀

---

### [2] downscale_frames.py — 批量縮圖

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `max_side` | 長邊最大像素 | 1000 |
| 插值方式 | INTER_AREA | — |

**為什麼需要縮圖：**
- COLMAP 特徵提取計算量與解析度平方成正比
- 1000px 在折床機這種「室內固定場景」下精度已足夠
- 縮圖不影響相機姿態精度，只影響點雲密度

---

### [3] sfm_colmap.py — SfM 三維重建

| 匹配策略 | 說明 | 適用場景 |
|---------|------|---------|
| `sequential_matcher` | 相鄰幀互匹配 | 影片（本 pipeline 預設） |
| `spatial_matcher` | 空間鄰近幀匹配 | 已知 GPS 座標的大場景 |

**COLMAP 輸出的核心資料:**

```
cameras.bin  → 鏡頭內參（焦距 f, 畸變係數 k1/k2）
               → 用途：3DGS 訓練、相機標定（T03）

images.bin   → 每幀影格的 6-DoF 相機姿態
               → 位置 (X, Y, Z) + 方向 (四元數)
               → 用途：直接作為 Splat the Net [100] 的輸入

points3D.bin → 場景稀疏 3D 點（特徵點三角化）
               → 用途：確認重建品質、匯出 PLY
```

---

### [4] export_unity_pytorch.py — 格式匯出

| 輸出格式 | 用途 |
|---------|------|
| `scene.ply` | Unity 場景 — 折床機幾何骨架 |
| `cameras.txt` | 3DGS（Splat the Net）/ NeRF 訓練輸入 |

---

## 四、對研究 T 題的貢獻

| T 題 | 問題 | 本 pipeline 的解答 |
|------|------|-------------------|
| **T12** | Unity 合成資料 domain gap | 從真實拍攝建幾何，不是手工猜，根本縮小 gap |
| **T03** | 多相機 rig 標定 | COLMAP `images.bin` 直接給出相機姿態，不需要額外標定板 |
| **T02** | 3DGS 建圖方案 | `cameras.txt` + `frames_1k/` 直接是 3DGS 訓練的標準輸入 |
| **T13** | 最小標注集 | Unity 場景幾何準確 → 合成資料品質高 → 所需真實圖更少 |
| **B06** | 冷啟動零標注資料 | 精確 Unity 場景 → 大量合成標注資料 → 零現場資料可訓練起步模型 |

---

## 五、接下來的步驟（按順序）

### Step 1：跑 Dense 重建（你已有腳本）

```bash
# 你的 dense/ 目錄已有 patch-match.cfg 和 fusion.cfg
colmap patch_match_stereo --workspace_path data/work
colmap stereo_fusion --workspace_path data/work --output_path exports/fused.ply
```

**結果：** `fused.ply` → 密集點雲 → Unity 裡看到「面」而非「點」

---

### Step 2：處理 Scale 問題（重要！）

COLMAP 輸出的是**相對尺度**，不是公尺單位。需要一個比例參考：

**方法 A（推薦）：拍攝前放一個已知尺寸物件**
```
在折床機前放一塊 A4 紙（210 × 297mm）或量好的棋盤格
→ COLMAP 重建後，量 A4 紙在點雲中的像素距離
→ 算出 scale factor（1 COLMAP unit = X mm）
→ 在 Unity 匯入時套用 scale
```

**方法 B（事後量）：**
```python
# 從 points3D.bin 中選兩個已知距離的點
# 算出 scale factor 後校正整個場景
```

---

### Step 3：COLMAP → Unity 精確場景

```
exports/scene.ply（或 fused.ply）
    │
    ▼ Unity 匯入流程：
    1. 安裝 Unity 2022 LTS + Perception package
    2. Import → PLY → 設定 scale factor（Step 2 算出）
    3. 場景已對齊真實折床機幾何
    4. 加入代表性工件模型（從 DXF 自動生成或手工建）
```

---

### Step 4：COLMAP → 3DGS 訓練（Phase-0A，桌機先跑）

**策略：先 Phase-0A（靜態底圖）→ 再 Phase-0B（動態追蹤）**

```
data/frames_1k/（影格）
    +
data/work/sparse/0/（COLMAP binary，直接兼容，不需轉換）
    │
    ▼ gsplat（Berkeley 加速版，訓練快 2-3×，記憶體 -40%）
    → RTX 5070 Ti 訓練時間：∼20-40 min / 場景
    → 輸出 exports/3dgs/scene.splat（靜態折床機底圖）
    → Phase-0B 的 Deformable-3DGS 以此為初始化，只追形變差量
```

**Phase-0A 關鍵訓練參數：**

| 參數 | 值 | 說明 |
|------|----|----- |
| `--iterations` | 30000 | 5070 Ti 約 20 min |
| `--densify_until_iter` | 15000 | 點雲密化截止 |
| `--sh_degree` | 3 | 金屬反光補償（多一階球諧函數）|
| `--position_lr_max_steps` | 30000 | 位置學習率衰減步數 |

---

## 六、系統整體架構（兩層系統）

```
【第1層】生產工廠 + 統一輸出
c:\3d-recon-pipeline/
    ├── data/frames_1k/              ← COLMAP 輸入（縮圖影格）
    ├── data/work/sparse/0/          ← COLMAP 輸出（稀疏點雲 + 相機姿態）
    ├── exports/3dgs/
    │   ├── ckpts/ckpt_*.pt          ← gsplat 模型 checkpoint
    │   ├── ply/point_cloud_final.ply
    │   ├── renders/                 ← 驗證渲染圖
    │   └── stats/                   ← 訓練指標
    └── exports/L5D_reports/         ← AGENT 報告輸出

        ⬇️ 工作流

【第2層】質檢系統
d:\agent_test/
    ├── agents/
    │   ├── analyzer/        讀深度圖、Alpha、計算場景尺寸
    │   ├── critic/          兼容性檢查（Unity 要求）
    │   └── doc_agent/       輸出評估報告
    └── 讀取路徑：c:\3d-recon-pipeline\exports\3dgs\
        ├── checkpoint
        ├── renders 驗證圖
        ├── PLY 檔案
        └── 統計數據
```

---

## 六-A、Phase-0 完整流程（詳細）

```
手機繞機台拍攝
    │
    ▼
本 Pipeline（COLMAP SfM）
    │
    ├── PLY 點雲 ──────────────────────────────────→ Unity 精確場景
    │                                                      │
    │                                               生成合成標注資料
    │
    ├── sparse/0/（binary）
    │       │
    │       ├── gsplat ─────────────────→ scene.splat（Unity 底圖）
    │       │  [RTX 5070 Ti, ~20 min]               │
    │       │                          Deformable-3DGS [Phase-0B]
    │       │                               （只追形變差量）
    │       │
    │       └── 2DGS ─────────────────→ scene.obj（工業件 mesh）
    │          [RTX 5070 Ti, ~25 min]       │
    │                                  工件尺寸驗證（< 2 mm）
    │                                  scale 校正依據
    │                                  Unity 碰撞體處理
    │
    └── cameras.txt ───────────────────────────────→ Splat the Net [100]
                                                    （增量更新，最終部署）
                                                           │
                                                    Jetson 邊緣推理

---

## 六-B、影片抽幀後的語義 ROI bootstrap（L0-S2 備選）

若後續要把 `L0` 從純 `OpenCV + NumPy` 提升到語義 ROI 路線，正式做法不應直接從全量影格微調 `YOLO11-seg`，而應先做少量 bootstrap 標註。

### 目的

- 驗證「語義 ROI」是否比 heuristic ROI 更穩
- 降低直接投入大規模標註的風險
- 作為 `Grounded-SAM-2 / zero-shot ROI` 之後的第二階段

### 第一輪 bootstrap 規格

- 類別數：`1`
- 類別名稱：`punch_holders`
- 張數：`20 ~ 30` 張，最多 `50` 張
- 來源：現有影片抽幀 `frames_1600`

### 抽樣原則

三種畫面都必須涵蓋：

1. 主體清楚、正面視角
2. 側面 / 遠視角
3. 背景雜、反光重、較難的畫面

### 標註原則

- 第一輪允許粗 segmentation / 粗 polygon
- 不追求 pixel-perfect 邊界
- 只標可見的 `punch_holders`
- 明顯背景不要包進來太多
- 線材、風扇、工具、桌面、手套、螢幕先全部視為背景
- 若影格內沒有 `punch_holders`，保留空標註作為負樣本

### 正式判讀

- 這批圖雖然背景雜，但仍可做第一輪 bootstrap
- 問題不在於「能不能標」，而在於是否值得大規模精標
- 在未證明語義 ROI 有訊號前，不應直接投入完整 YOLO segmentation dataset
- 正確順序應為：
  1. zero-shot / prompt-based ROI 驗證
  2. 少量 bootstrap 粗標
  3. 若有訊號，再擴充成正式 YOLO11-seg 訓練集

### 目前已完成的 bootstrap 結果

- `bootstrap_12_split`
  - split 驗證：`Box mAP50 = 0.665`、`Mask mAP50 = 0.665`
  - 全量 inference：`59 / 853`
- `bootstrap_24_split`
  - split 驗證：`Box mAP50 = 0.83`、`Mask mAP50 = 0.83`
  - 但全量 inference 在 `conf=0.25` 下為 `0 / 853`
  - 判讀：split 指標過度樂觀，泛化失敗
- `bootstrap_36_split`
  - split 驗證：`Box mAP50 = 0.746`、`Mask mAP50 = 0.746`
  - `Precision = 0.935`、`Recall = 0.750`
  - 全量 inference confidence sweep：
    - `conf=0.25`：`724 / 853`
    - `conf=0.15`：`765 / 853`
    - `conf=0.10`：`781 / 853`
    - `conf=0.05`：`806 / 853`

### 目前正式結論

- `punch_holders` 單類別 semantic ROI 已證明可學到、可在全量影格上產生大量有效偵測
- 這條路線已不是弱訊號，可作為 `L0-S2` 正式候選
- 但它接回 `L0` 後，在 Gate 2 `5000 iter` 短訓練仍未打敗 heuristic ROI
- 因此目前定位應為：
  - `bootstrap / semantic ROI` 可繼續保留
  - 但尚未升格為正式主線

### 後續公平對照與最新收斂

### 重要更正：`train_3dgs.py` scene dir 污染

2026-04-18 後續檢查發現，先前部分 `H_D0` 相關訓練雖然命令列指定了：

- `imgdir = H_D0 selected_frames`
- `colmap = H_D0 sparse model`

但舊版 `train_3dgs.py` 會重用全域 [data/colmap_scene](C:/3d-recon-pipeline/data/colmap_scene)，若該 junction 已存在，就不會依當輪輸入重建 scene dir。

實際後果：

- 部分本應是 `143 張 H_D0` 的訓練，trainer log 顯示：
  - `[Parser] 853 images`

因此以下舊結論暫時失效：

- 舊版 `H_D0 full train`
- 舊版 `H_D0 shorttrain`
- 舊版 `masked H_D0 shorttrain`
- 舊版 `masked H_D0 full train` 早期判讀

修正方式已完成：

- `train_3dgs.py` 現在每輪會在輸出目錄下建立獨立 `_colmap_scene`
- 不再重用全域 [data/colmap_scene](C:/3d-recon-pipeline/data/colmap_scene)
- 新的正式規則是：
  1. 每輪 `Phase 1B` 都要檢查 `[Parser] N images`
  2. `N` 必須等於本輪預期影格數
  3. 不符合者視為污染 run，不得進正式結論

修正後的新 run 已完成，且已確認：

- `masked H_D0 full train (fixed)` 的 log 內有：
  - `[Parser] 143 images`
- `plain H_D0 full train (fixed)` 的 log 內也有：
  - `[Parser] 143 images`

在 `L0 2x2` 收斂後，舊版曾補做 `H_D0` 的公平 full train：

- `H_D0`
  - `143` 張最佳子集
  - `30000 iter`
  - `PSNR 25.3347`
  - `SSIM 0.8710`
  - `LPIPS 0.20556`
  - `num_GS 682,669`
- `U_base`
  - `PSNR 25.3986`
  - `SSIM 0.8718`
  - `LPIPS 0.2049`
  - `num_GS 684,641`

舊版判讀（目前失效）：

- `H_D0` 沒有超過 `U_base`
- 但已非常接近 `U_base`
- 這證明 `143` 張最佳子集不是死路
- 也證明先前 `5000 iter` 的 `0.265x` 不能直接拿來判死

修正後正式結果：

- `plain H_D0 full train (fixed)`
  - `Run: H_D0_fulltrain_FIXED_20260418_114253`
  - `PSNR 22.2590`
  - `SSIM 0.8125`
  - `LPIPS 0.24772`
  - `num_GS 504,973`

正式判讀：

- 舊版 `H_D0 ≈ U_base` 的結論撤回
- 在真正乾淨的 `143 張 regime` 下，`plain H_D0` 明顯劣於 `U_base`

之後已補做 `machine-level loss mask` smoke（以下為舊版污染紀錄）：

- 規則：`train_3dgs.py --loss-mask-dir`
- mask 語義：`PNG 非零像素 = 排除區域（背景）`
- 來源：heuristic 機台 ROI 生成的整機級背景遮罩

`H_D0` 的 `5000 iter` 結果：

- plain
  - `PSNR 22.9412`
  - `SSIM 0.83117`
  - `LPIPS 0.26545`
  - `num_GS 379,699`
- masked
  - `PSNR 22.9644`
  - `SSIM 0.83186`
  - `LPIPS 0.26538`
  - `num_GS 376,537`

舊版判讀（目前失效）：

- `machine-level loss mask` 比直接用 `punch_holders` mask 更合理
- 方向正確，且結果略優於 plain
- 但收益仍很小，目前只能算弱正訊號

修正後正式結果：

- `plain H_D0 full train (fixed)`
  - `PSNR 22.2590`
  - `SSIM 0.8125`
  - `LPIPS 0.24772`
  - `num_GS 504,973`
- `masked H_D0 full train (fixed)`
  - `Run: H_D0_masked_fulltrain_FIXED_20260418_111512`
  - `PSNR 22.0658`
  - `SSIM 0.8067`
  - `LPIPS 0.25456`
  - `num_GS 470,565`

正式判讀：

- 在公平的 `143 張 fixed` 對照下，`masked` 明顯差於 `plain`
- 目前這版 `machine-level loss mask` 應視為負面因素
- 因此這條線暫停，不再當前優先方向

因此目前這一段的正式收斂為：

- `semantic ROI` 已可用，但仍未升為正式主線
- `H_D0 = heuristic ROI + duplicate penalty off` 仍是目前 `L0` 最佳配置
- `L0 2x2` 的 Gate 1 幾何結論仍有效
- 舊版 `H_D0 / masked H_D0` 訓練結論已被 fixed rerun 覆寫
- 目前不再細磨 `L0` 選幀，也不繼續這版 `machine-level loss mask`

之後補做了兩組 train 端 probe，同樣全部基於：

- `H_D0 selected_frames = 143 張`
- 獨立 `_colmap_scene`
- `[Parser] 143 images`

公平基準：

- `plain H_D0 @ val_step4999`
  - `PSNR 19.6872`
  - `SSIM 0.7328`
  - `LPIPS 0.39939`

`app_opt=True` smoke：

- `Run: app_opt_h_d0_smoke_20260418_124511`
- `PSNR 19.4702`
- `SSIM 0.7460`
- `LPIPS 0.35854`

`sh_degree=1` smoke：

- `Run: sh1_h_d0_smoke_20260418_132355`
- `PSNR 20.1197`
- `SSIM 0.7553`
- `LPIPS 0.35909`

`sh_degree=1 + app_opt=True` smoke：

- `Run: sh1_appopt_h_d0_smoke_20260418_134418`
- `PSNR 19.4501`
- `SSIM 0.7458`
- `LPIPS 0.35668`

smoke 結論：

- 三個 probe 在 `5000 iter` 都比 `plain H_D0 @ step4999` 好
- 但 full train 才是正式證據，所以只把 `app_opt` 和 `sh_degree=1` 往前推

`app_opt=True` full train：

- `Run: app_opt_h_d0_fulltrain_20260418_124945`
- `PSNR 20.0509`
- `SSIM 0.7669`
- `LPIPS 0.27352`

`sh_degree=1` full train：

- `Run: sh1_h_d0_fulltrain_20260418_135309`
- `PSNR 21.8149`
- `SSIM 0.8017`
- `LPIPS 0.25826`

對照 `plain H_D0 full train (fixed)`：

- `plain H_D0 full train (fixed)`
  - `PSNR 22.2590`
  - `SSIM 0.8125`
  - `LPIPS 0.24772`
- `app_opt=True`
  - 明顯更差
- `sh_degree=1`
  - 優於 `app_opt`，但仍明顯更差

正式收斂更新：

- `machine-level loss mask`：負面因素
- `app_opt`：full train 下為負面因素
- `sh_degree=1`：full train 下仍劣於 `plain H_D0 (fixed)`
- 因此目前 `143 張 regime` 的最佳結果仍是：
  - `plain H_D0 full train (fixed)`
- 現階段 train 端 probe 應暫停；若未來重啟，優先補：
  - 固定公平 eval / validation protocol
  - 可與 `U_base` 對齊的 validation views
  - 再考慮 `pose_opt`

之後補做 `853 張 U_base regime` 的中期 probe（`7000 iter`）：

- `plain`
  - `PSNR 23.445`
  - `SSIM 0.8420`
  - `LPIPS 0.250`
  - `num_GS 429,942`
- `app_opt=True`
  - `PSNR 22.651`
  - `SSIM 0.8299`
  - `LPIPS 0.259`
  - `num_GS 417,085`
- `sh_degree=1`
  - `PSNR 23.209`
  - `SSIM 0.8370`
  - `LPIPS 0.256`
  - `num_GS 423,690`
- `opacity_reg=0.01`
  - `PSNR 23.055`
  - `SSIM 0.8336`
  - `LPIPS 0.267`
  - `num_GS 316,312`
- `pose_opt=True`
  - `PSNR 23.297`
  - `SSIM 0.8391`
  - `LPIPS 0.251`
  - `num_GS 427,813`

`853 張` 中期 probe 判讀：

- 最佳仍是 `plain`
- `app_opt` 不成立
- `sh_degree=1` 也不成立
- `opacity_reg=0.01` 明顯更差
- `pose_opt=True` 非常接近 `plain`，但仍略差

注意：

- [train_3dgs.py](C:/3d-recon-pipeline/src/train_3dgs.py) 現已 expose：
  - `opacity_reg`
  - `pose_opt`
- 但在目前 `853 張 U_base` 主線的 `7000 iter` probe 中，這兩個方向都沒有打贏 `plain`。

之後補做 `MCMCStrategy`：

`7000 iter` 中期 probe：

- `Run: u_base_mcmc_mid_probe_20260418_151718/mcmc`
- `PSNR 24.107`
- `SSIM 0.8569`
- `LPIPS 0.224`
- `num_GS 1,000,000`

相對 `plain U_base @ 7000 iter`：

- `PSNR +0.662`
- `SSIM +0.0149`
- `LPIPS -0.026`

`30000 iter` full train：

- `Run: u_base_mcmc_fulltrain_20260418_155339/mcmc`
- `PSNR 26.1572`
- `SSIM 0.8826`
- `LPIPS 0.19187`
- `num_GS 1,000,000`

相對正式 baseline `U_base`：

- `PSNR +0.7586`
- `SSIM +0.0108`
- `LPIPS -0.01306`

正式結論更新：

- `MCMCStrategy` 是目前第一個正式打贏 `U_base` 的方向
- 但高斯數量直接撞到 `1,000,000` 上限，成本明顯增加
- 因此後續重點從「找下一個 train 小參數」轉為：
  - `MCMC` 的 export / Unity 成本檢查
  - 若有需要，再做 `cap_max` 或其他成本控制 follow-up

補充：

- 目前本地 [gsplat_runner/simple_trainer.py](C:/3d-recon-pipeline/gsplat_runner/simple_trainer.py) 的 `mcmc` 不是單一策略旗標，而是官方 preset bundle：
  - `strategy = MCMCStrategy`
  - `init_opa = 0.5`
  - `init_scale = 0.1`
  - `opacity_reg = 0.01`
  - `scale_reg = 0.01`
- 所以現在正式成立的結論是：
  - **官方 `mcmc preset` 打贏了 `default preset`**
  - 不是單獨某個 train 小參數翻盤

後續若要再做策略層 follow-up，優先順序應是：

1. `MCMC + cap_max / 成本控制`
2. `MCMC + antialiased`
3. 若明確追求更低 `LPIPS`，再考慮更高 `cap_max`

目前不建議優先重啟或與 `MCMC` 盲目組合的方向：

- `app_opt`
- `sh_degree=1`
- `pose_opt`
- `machine-level loss mask`
- `143 張 H_D0 regime`
                                                           │
                                                    Phase-1 系統啟動
```

---

## 七、已知限制與後續工作

| 限制 | 說明 | 解決方向 |
|------|------|---------|
| **動態物件** | 折彎中的工件屬於動態，COLMAP 可能重建失敗 | 先拍靜態場景（無工件），工件另外單獨掃描 |
| **金屬反光** | E01 金屬表面反光導致特徵點提取失敗 | 使用消光噴霧或黏貼 AR Marker 輔助 |
| **Scale 不確定** | COLMAP 尺度任意 | 參見 Step 2 比例校正 |
| **室內光線不均** | 影響特徵匹配 | 盡量在白天均勻光線下拍攝 |

---

## 八、相依套件與環境

### Python 環境（已建立 .venv）

```
環境類型：venv
Python：3.12.10
位置：c:\3d-recon-pipeline\.venv
```

| 套件 | 版本 | 說明 |
|------|------|------|
| numpy | 2.3.5 | |
| opencv-python | 4.13.0.92 | |
| tqdm | 4.67.3 | |
| typer[all] | 0.24.1 | |
| rich | 14.3.3 | |
| Pillow | 12.0.0 | |
| torch | 2.12.0.dev+cu128 | nightly，支援 sm_120 |
| torchvision | 0.26.0.dev+cu128 | nightly |
| ninja | 1.13.2 | 編譯 CUDA extension 用 |

### 外部工具

| 工具 | 版本 | 狀態 | 安裝方式 |
|------|------|------|----------|
| COLMAP | — | ✅ 已安裝 | `C:/tools/colmap/bin/colmap.exe` |
| ffmpeg | — | — | 加入 PATH 即可 |
| ninja | 1.13.2 | ✅ 已安裝 | `winget install Ninja-build.Ninja` |
| CUDA Toolkit | 13.1 | ⏳ 安裝中 | `winget install Nvidia.CUDA` |
| VS2022 C++ | 14.44 | ✅ cl.exe 已就緒 | VS Installer 修改工作負載 |
| gsplat | 1.5.3 | ✅ JIT 已編譯 | 見 §十一（Windows 全流程） |

### 執行全流程

```bash
# 啟動虛擬環境
.venv\Scripts\Activate.ps1

# 完整 pipeline（步驟 1-4）
python -m src.pipeline

# 或分步執行
python -m src.extract_frames data/video/factory.mp4
python -m src.downscale_frames
python -m src.sfm_colmap --imgdir data/frames_1k --work data/work
python -m src.export_unity_pytorch
```

---

## 九、與論文研究的對應關係

本 pipeline 在論文中的定位是：

> **Phase-0 真實環境數位化工具**，解決 B06 冷啟動問題的第一步，  
> 同時為 T02（3DGS）、T03（相機標定）、T12（sim-to-real）提供資料基礎。

對應文獻：
- [70] D-REX：real-to-sim 思路（本 pipeline 實現了其核心精神）
- [100] Splat the Net：直接消費本 pipeline 的 COLMAP 輸出
- [103] Geometric Look-Angle Shaping：用 COLMAP 相機姿態決定最優安裝位置

---

## 十、3DGS 整合規劃（Phase-0A → Phase-0B）

### 策略總覽

| 階段 | 方法 | 工具 | 目標 | 驗收指標 |
|------|------|------|------|----------|
| **Phase-0A 底圖** | gsplat 靜態建圖 | RTX 5070 Ti | 折床機 Unity 底圖 | PSNR > 25 dB，>5 fps Jetson |
| **Phase-0A 幾何** | **2DGS 工業件重建** | RTX 5070 Ti | 工件 mesh 尺寸校正 | **重建誤差 < 2 mm** |
| **Phase-0.5** | 2DGS + Mip-Splatting | RTX 5070 Ti | 金屬件深度補全 | 邊緣誤差 < 1 mm |
| **Phase-0B** | Deformable-3DGS | RTX 5070 Ti（訓）+ Jetson（推） | 折彎動態追蹤 | HOLD_OK 前後形變捕捉 |

### gsplat vs 2DGS 各司其職

```
gsplat（nerfstudio-project, Berkeley+NVIDIA+Meta）
  → 4x 省記憶體，15% 指令快，已整合 NVIDIA 3DGUT（Apr 2025）
  → 學習目標：規登novew view synthesis（看起來好）
  → 用途 A：Unity 場景底圖 + Jetson 還彩器

2DGS（hbb1, SIGGRAPH 2024）
  → Surfel（2D 圓餅）表示，幾何精度最佳，可導出 .obj mesh
  → 學習目標：幾何准確（尺寸符合）
  → 用途 B：工件尺寸驗證（scale 校正依據）+ Unity 碰撞體
  → 相同 COLMAP 輸入格式，可與 gsplat 頁行訓練

扑克：5070 Ti 同時跑兩個，各取所限。
```

### Phase-0A 環境安裝

> ⚠️ **RTX 5070 Ti (sm_120/Blackwell) 注意事項**  
> gsplat 官方 wheel 無 Windows 版本，且 sm_120 支援需 CUDA 12.8+。  
> 詳細解決方案見 §十一。

```bash
# ① gsplat（場景底圖 / Jetson 還彩）
# 路線 A：降 PTX 編譯（最穩，效能 ~95%，不需 sm_120 原生支援）
$env:TORCH_CUDA_ARCH_LIST = "9.0+PTX"
pip install gsplat --no-build-isolation

# 路線 B：sm_120 原生（需 CUDA Toolkit 12.8+ 且從 source 編譯）
$env:TORCH_CUDA_ARCH_LIST = "9.0;12.0"
pip install gsplat --no-build-isolation

# ② 2DGS（工業件幾何 / mesh 導出 / scale 校正）
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
cd 2d-gaussian-splatting
conda env create -f environment.yml
conda activate surfel_splatting

# 訓練（和 3DGS 相同 COLMAP 輸入）
python train.py -s data/work/ --source_path data/frames_1k/

# 導出 mesh（工業件尺寸檢驗）
python render.py -m output/scene/ -s data/work/ --mesh_res 1024
```

### Phase-0A 與現有 Pipeline 的關係

```
現有 4 個腳本（不需修改）：
  extract_frames.py
  downscale_frames.py
  sfm_colmap.py            → 輸出 data/work/sparse/0/
  export_unity_pytorch.py  → 輸出 cameras.txt

新增第 5 個腳本 ✅ 已建立：
  scale_calibrate.py       → 從 points3D.bin 計算 1 COLMAP unit = X mm
    模式：--mode list      列出所有低誤差點
    模式：--mode a4        用 A4 紙四角點計算 scale
    模式：--mode ruler     指定兩點+真實距離計算 scale
    輸出：exports/scale.json

新增第 6 個腳本 ✅ 已建立：
  train_3dgs.py            → gsplat wrapper（Phase-0A）
    輸入：data/work/sparse/0/   ← COLMAP binary 直接兼容
    輸入：data/frames_1k/       ← 已縮圖影格
    輸入：exports/scale.json    ← 可選，套用公尺 scale
    輸出：exports/3dgs/         ← .ply + splats.pth
    說明書參數：--iterations 30000 --sh-degree 3 --densify-until 15000
```

### Phase-0A → Phase-0B 銜接機制

> **關鍵設計：B 不從零訓練，以 A 的靜態底圖初始化**

```
Phase-0A 輸出：
  exports/3dgs/scene.splat       ← 靜態折床機底圖（含公尺 scale）
  exports/3dgs/cameras.json      ← 相機標定完整版

Phase-0B 接手：
  Deformable-3DGS
    → 讀 scene.splat 作為 Gaussian 初始化
    → 折彎前後只訓練 delta_xyz 差量
    → 訓練時間：∼5 min（而非從零的 40 min）
    → 這是 Splat the Net [100] 增量更新的工程實作
```

### 金屬反光處理路線（E01）

```
1. 拍攝期：貼 Aruco Marker 輔助特徵提取
2. LingBot-Depth [17]：補全金屬件缺失深度（3DGS 前處理）
3. 訓練期：sh_degree=3（比預設多，補償多視角顏色不一致）
4. 備選：Mip-Splatting（抗混疊，金屬邊緣更清晰）
```

### Scale 校正與 3DGS 整合

```python
# scale_calibrate.py ✅ 已建立於 src/scale_calibrate.py
# 用法（先列出低誤差點挑選 A4 角點）：
#   python -m src.scale_calibrate --mode list --max-err 1.0
#   python -m src.scale_calibrate --mode a4 --ids 10 20 30 40
# 輸出：exports/scale.json → {scale_mm_per_unit, scale_m_per_unit}

A4_width_mm  = 210.0
A4_height_mm = 297.0
# A4 對角線 = sqrt(210² + 297²) ≈ 363.8 mm
# scale_factor = 363.8 / colmap_dist（COLMAP 中量出的對角線長度）
```

---

*說明書版本 v1.4 — 2026-03-05 §十一 gsplat Windows 全流程完成；新增 §十二 Phase-0A 訓練紀錄*

---

## 十二、Phase-0A 首次訓練紀錄（2026-03-05）

### 環境摘要

| 項目 | 值 |
|------|---|
| GPU | RTX 5070 Ti（sm_120，Blackwell） |
| CUDA Toolkit | 13.1（透過 `C:\cuda13` junction） |
| PyTorch | 2.12.0.dev20260304+cu128（nightly） |
| gsplat | 1.5.3（JIT 編譯，`gsplat_cuda.pyd`） |
| 影格數 | 1088 張（`data/frames_1k/*.png`） |
| COLMAP 點數 | 81,654 點（`data/work/sparse/0/points3D.bin`） |

### 執行指令

```powershell
# 設定 VS + CUDA 環境（每次新 terminal 都需執行）
$vcvarsall = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$envLines = cmd.exe /c "`"$vcvarsall`" x64 && set" 2>&1
foreach ($ln in $envLines) { if ($ln -match "^([^=]+)=(.*)$") { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process") } }
$env:CUDA_HOME = "C:\cuda13"; $env:CUDA_PATH = "C:\cuda13"; $env:PATH = "C:\cuda13\bin;" + $env:PATH
$env:TORCH_CUDA_ARCH_LIST = "9.0+PTX"; $env:DISTUTILS_USE_SDK = "1"

# 執行訓練
python -m src.train_3dgs
# 等同於：
# python -m src.train_3dgs --imgdir data/frames_1k --colmap data/work/sparse/0 \
#   --outdir exports/3dgs --iterations 30000 --sh-degree 3
```

### 訓練速度

| 階段 | 速度 | 說明 |
|------|------|------|
| Phase-0A  開始（SH 0 階） | ~126 it/s | 初期點少，速度最快 |
| SH 升至 3 階後 | ~47 it/s | Densification 後點雲增加，正常下降 |
| gsplat JIT 首次編譯 | 109.87 秒 | 30 個 CUDA 檔並行，後續快取 |

### 重要路徑記錄

```
訓練輸出目錄：
  exports/3dgs/

gsplat CUDA 快取（首次編譯後快取，之後秒載入）：
  %USERPROFILE%\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cu128\gsplat_cuda\

colmap_scene 目錄結構（junction）：
  data/colmap_scene/
    images/   → data/frames_1k/       （junction）
    sparse/
      0/      → data/work/sparse/0/   （junction）
```

### 常見問題排查

| 錯誤訊息 | 原因 | 解法 |
|---------|------|------|
| `ModuleNotFoundError: No module named 'gsplat'` | pip.exe 指向舊路徑 | 改用 `python -m pip install gsplat==1.5.3` |
| `ninja: fatal: ReadFile: The handle is invalid` | VS Code ConPTY stdin 問題 | 套用補丁 2（`stdin=subprocess.DEVNULL`） |
| `CreateProcess failed: The system cannot find the file specified` | shlex 單引號包裹 nvcc 路徑 | 套用補丁 3（`subprocess.list2cmdline`） |
| `CUDA versions mismatch` | torch cu128 vs CUDA Toolkit 13.x | 套用補丁 1（改為 warning） |
| `cl: D8021 '/Wno-attributes'` | MSVC 不支援 GCC flag | 套用補丁 4（`_backend.py`） |
| `invalid combination of type specifiers: bool char` | `rpcndr.h` `#define small char` 衝突 | 套用補丁 4（加 `-DWIN32_LEAN_AND_MEAN`） |

---

## 十一、RTX 5070 Ti (sm_120) 安裝實錄

> 更新：2026-03-05

### 問題背景

RTX 5070 Ti 屬於 Blackwell 架構（sm_120），而 3DGS 相關框架的 CUDA extension 普遍尚未原生支援：

| 框架 | Issue | 狀態 |
|------|-------|------|
| gsplat (nerfstudio) | [#855](https://github.com/nerfstudio-project/gsplat/issues/855), [#866](https://github.com/nerfstudio-project/gsplat/issues/866) | 🔴 Open，無官方修復 |
| 原版 3DGS (graphdeco) | [#1303](https://github.com/graphdeco-inria/gaussian-splatting/issues/1303), [#1215](https://github.com/graphdeco-inria/gaussian-splatting/issues/1215) | 🟡 Open，有成功案例 |

**根本原因：** sm_120 原生支援需要 CUDA 12.8+，而多數框架 wheel 最高只編到 sm_90。

### 兩條解決路線

| | 路線 A（PTX 降編）| 路線 B（原生 sm_120）|
|---|---|---|
| nvcc 版本 | 任何 ≥ 12.x | 必須 ≥ 12.8 |
| 效能 | ~95%（JIT PTX → sm_120）| 100% |
| 編譯指令 | `TORCH_CUDA_ARCH_LIST=9.0+PTX` | `TORCH_CUDA_ARCH_LIST=9.0;12.0` |
| 前置條件 | VS2022 C++ + 任意 CUDA Toolkit | VS2022 C++ + CUDA Toolkit 12.8+ |

**本專案選擇：路線 A（先跑通，後期可升路線 B）**

### 安裝步驟記錄（2026-03-05 實際執行）

#### Step 1 — PyTorch cu128 nightly（已完成 ✅）

```powershell
# 安裝支援 sm_120 的 PyTorch（nightly cu128）
pip install --pre torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall --no-deps
# 結果：torch-2.12.0.dev20260304+cu128  torchvision-0.26.0.dev+cu128
# GPU available: True（sm_120 偵測正常，有 UserWarning 但可忽略）
```

#### Step 2 — ninja（已完成 ✅）

```powershell
winget install --id Ninja-build.Ninja -e --accept-source-agreements --silent
# 結果：ninja 1.13.2 安裝至 PATH
```

#### Step 3 — VS2022 C++ 工作負載（已完成 ✅）

```powershell
# VS Installer 靜默安裝（需管理員權限）
Start-Process "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" \
  -ArgumentList 'modify --installPath "C:\Program Files\Microsoft Visual Studio\2022\Community" \
  --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --quiet --norestart' \
  -Verb RunAs -WindowStyle Hidden
# 結果：cl.exe 14.44.35207 就緒
#   位置：C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
```

#### Step 4 — CUDA Toolkit（已完成 ✅）

```powershell
# winget 安裝最新 CUDA Toolkit
winget install --id Nvidia.CUDA -e --accept-source-agreements --silent
# 結果：CUDA Toolkit 13.1 安裝至預設路徑
#   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe

# ⚠️ CUDA 13.1 路徑含空格 → 建立 junction 避免 ninja 路徑問題
cmd.exe /c "mklink /J C:\cuda13 `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`""
```

#### Step 5 — gsplat Windows 編譯全流程（已完成 ✅）

> **實際耗時：CUDA JIT 首次編譯約 110 秒（30 個 CUDA 文件並行）**

##### 問題根因分析

Windows 上 `pip install gsplat` 有四個互相獨立的阻塞問題：

| # | 症狀 | 根本原因 | 修復位置 |
|---|------|---------|----------|
| 1 | `RuntimeError: CUDA versions mismatch` | torch nightly cu128 vs CUDA Toolkit 13.1 版本比對過嚴 | `cpp_extension.py` 行 567 |
| 2 | `ninja: fatal: ReadFile: The handle is invalid` | VS Code ConPTY 給 ninja 無效 stdin 句柄 | `cpp_extension.py` `_run_ninja_build()` |
| 3 | `CreateProcess failed: The system cannot find the file specified` | `shlex.join()` 對路徑加 POSIX 單引號 → Windows 把 `'C:\cuda13\bin\nvcc'` 當字面檔名 | `cpp_extension.py` 行 3069 |
| 4 | `cl: D8021 無效的數字引數 '/Wno-attributes'` | `-Wno-attributes` 是 GCC 選項，MSVC 不支援 | `gsplat/cuda/_backend.py` |
| 5 | `invalid combination of type specifiers: bool char` | Windows SDK `rpcndr.h` 定義 `#define small char`，衝突 torch header | `gsplat/cuda/_backend.py` |

##### 修補方案（共 4 個補丁）

**補丁 1 — CUDA 版本比對改為 warning（已在先前 session 完成）**

```python
# 檔案：.venv\Lib\site-packages\torch\utils\cpp_extension.py 行 ~567
# 原始：
raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(d, torch_version, cuda_str))
# 修改後：
warnings.warn(f"CUDA version mismatch (d={d} torch={torch_version} toolkit={cuda_str}), ignored")
```

**補丁 2 — ninja stdin 改為 DEVNULL（修正 VS Code ConPTY 無效句柄）**

```python
# 檔案：.venv\Lib\site-packages\torch\utils\cpp_extension.py 行 ~2807
# 在 subprocess.run() 加入 stdin=subprocess.DEVNULL
subprocess.run(
    command,
    shell=IS_WINDOWS and IS_HIP_EXTENSION,
    stdout=stdout_fileno if verbose else subprocess.PIPE,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,  # ← 新增：修正 ninja ReadFile invalid handle
    cwd=build_directory,
    check=True,
    env=env)
```

**補丁 3 — nvcc 路徑改用 Windows 雙引號（修正 CreateProcess 找不到檔案）**

```python
# 檔案：.venv\Lib\site-packages\torch\utils\cpp_extension.py 行 ~3069
# 原始（對所有平台都用 POSIX 單引號）：
nvcc = shlex.join(_wrap_compiler(nvcc))
# 修改後（Windows 用 subprocess.list2cmdline 產生雙引號）：
nvcc_parts = _wrap_compiler(nvcc)
if IS_WINDOWS:
    nvcc = subprocess.list2cmdline(
        nvcc_parts if isinstance(nvcc_parts, list) else [nvcc_parts])
else:
    nvcc = shlex.join(nvcc_parts)
```

**補丁 4 — gsplat _backend.py Windows 專用 CUDA flags**

```python
# 檔案：.venv\Lib\site-packages\gsplat\cuda\_backend.py
# 原始：
extra_cflags = [opt_level, "-Wno-attributes"]
extra_cuda_cflags = [opt_level]
if not NO_FAST_MATH:
    extra_cuda_cflags += ["-use_fast_math"]
# 修改後：
import sys as _sys
extra_cflags = [opt_level]
if not _sys.platform == "win32":
    extra_cflags.append("-Wno-attributes")   # GCC/Clang only，MSVC 不支援
extra_cuda_cflags = [opt_level]
if not NO_FAST_MATH:
    extra_cuda_cflags += ["-use_fast_math"]
if _sys.platform == "win32":
    # rpcndr.h 的 #define small char 會衝突 PyTorch header 的 bool small 參數名
    extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN", "-allow-unsupported-compiler"]
    extra_cflags     += ["/DWIN32_LEAN_AND_MEAN"]
```

##### 安裝指令

```powershell
# ⚠️ pip.exe 可能硬編碼舊 Python 路徑，必須用 python -m pip
# （可用 strings pip.exe | grep python 確認）

# 1. 設定環境
$vcvarsall = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$envLines = cmd.exe /c "`"$vcvarsall`" x64 && set" 2>&1
foreach ($ln in $envLines) {
    if ($ln -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}
$env:CUDA_HOME             = "C:\cuda13"   # ← 無空格 junction
$env:CUDA_PATH             = "C:\cuda13"
$env:PATH                  = "C:\cuda13\bin;" + $env:PATH
$env:TORCH_CUDA_ARCH_LIST  = "9.0+PTX"    # 路線 A：PTX 降編
$env:DISTUTILS_USE_SDK     = "1"           # SDK 整合必要
$env:MAX_JOBS              = "4"

# 2. 安裝 gsplat Python wheel（CUDA 延遲到第一次 import 時 JIT 編譯）
# --no-build-isolation 確保使用我們已修補的 cpp_extension.py
c:\3d-recon-pipeline\.venv\Scripts\python.exe -m pip install gsplat==1.5.3 \
    --no-build-isolation --force-reinstall --no-deps --no-cache-dir

# 3. 觸發 JIT 編譯（首次 import 自動編譯，約 110 秒）
# 需要在有 VS + CUDA 環境的 terminal 中執行
c:\3d-recon-pipeline\.venv\Scripts\python.exe -c "
import os; os.environ['VERBOSE']='1'
from gsplat.cuda._backend import _C
print('_C =', _C)
"
# 預期輸出：
# [1/30] C:\cuda13\bin\nvcc ... （無單引號！）
# [30/30] link.exe ... gsplat_cuda.pyd
# gsplat: CUDA extension has been set up successfully in 109.87 seconds.
# _C = <module 'gsplat_cuda' from '...py312_cu128\gsplat_cuda\gsplat_cuda.pyd'>
```

##### 編譯快取位置

```
%USERPROFILE%\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cu128\gsplat_cuda\
  gsplat_cuda.pyd    ← 主 DLL（下次 import 秒完成，不重新編譯）
  build.ninja
  *.o
```

> **注意：每次更換 Python 版本或 torch 版本後，快取目錄名稱（py312_cu128）會改變，需重新編譯。**

### 原版 3DGS (graphdeco) 的 source patch 備忘

若未來需要直接跑原版 3DGS（非 gsplat），需在編譯前修改兩個標頭檔：

```cpp
// submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
#include <cstdint>   // ← 加這行

// submodules/simple-knn/simple-knn.cu
#include <float.h>   // ← 加這行
```

參考：[graphdeco #1303（RTX 5070 + Ubuntu 成功案例）](https://github.com/graphdeco-inria/gaussian-splatting/issues/1303)

---

## 十二、Phase-0A 首次訓練結果（2026-03-05）

### 訓練摘要

| 項目 | 數值 |
|------|------|
| 訓練步數 | 30,000 steps（完整） |
| 最終 Loss | 0.028 |
| 球諧階數 | SH degree 3（step 10,000 後升階） |
| 總訓練時間 | **12 分 51 秒** |
| 最終速度 | ~38.88 it/s |
| Gaussian 數量 | **2,095,607 個** |

### 輸出檔案

```
exports/3dgs/
├── ply/
│   └── point_cloud_final.ply      ← 主要輸出（495.6 MB，標準 3DGS PLY 格式）
├── ckpts/
│   ├── ckpt_6999_rank0.pt         283.2 MB（Step 7000 checkpoint）
│   └── ckpt_29999_rank0.pt        471.7 MB（最終 checkpoint，含完整 splats）
├── renders/
│   ├── val_step6999_*.png         136 張驗證渲染圖（step 7000）
│   └── val_step29999_*.png        136 張驗證渲染圖（step 30000）
├── videos/
│   ├── traj_6999.mp4              Step 7000 飛越軌跡影片
│   └── traj_29999.mp4             最終飛越軌跡影片 ✅
├── stats/
│   ├── val_step6999.json          PSNR/SSIM/LPIPS 中期指標
│   └── val_step29999.json         PSNR/SSIM/LPIPS 最終指標
└── cfg.yml                        訓練設定記錄
```

### 匯出 .ply 說明

訓練預設不儲存 .ply（`--save-ply` 未傳入），改用獨立匯出腳本：

```powershell
# 從 checkpoint 匯出標準 3DGS .ply
c:\3d-recon-pipeline\.venv\Scripts\python.exe src\export_ply.py
# --ckpt exports/3dgs/ckpts/ckpt_29999_rank0.pt（預設）
# --out  exports/3dgs/ply/point_cloud_final.ply（預設）
```

腳本位置：`src/export_ply.py`

---

## 十三、Unity 3DGS 匯入設定（Phase-0A → Unity）

### 工具鏈

| 元件 | 版本 / 路徑 |
|------|------------|
| Unity Editor | 6000.3.9f1（`C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe`） |
| Gaussian Splatting 插件 | [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)（package.json 內部名稱：`org.nesnausk.gaussian-splatting`） |
| 目標 Unity 專案 | `C:\Users\User\Downloads\phase0\Unity\BendViewer`（現有 URP 專案） |
| 輸入 .ply | `exports/3dgs/ply/point_cloud_final.ply`（495.6 MB，2,095,607 Gaussians） |

### 產出文件

| 檔案 | 說明 |
|------|------|
| `Assets/Editor/BatchGaussianImport.cs` | Batch 匯入 .ply → GaussianSplatAsset（6 個 .asset/.bytes 檔） |
| `Assets/Editor/BatchCreateScene.cs` | Batch 建立 FactoryGaussian.unity 場景，含 GaussianSplatRenderer |
| `Assets/GaussianAssets/point_cloud_final.asset` | 主 GaussianSplatAsset（~96 MB 壓縮） |
| `Assets/Scenes/FactoryGaussian.unity` | 完成場景，含 Camera + GaussianSplatRenderer |
| `unity_setup/create_scene_log.txt` | 最後一次 batch 執行完整 log |

### 步驟一：設定 manifest.json（⚠️ 必須無 BOM、正確 package 名稱）

```powershell
$PROJ = "C:\Users\User\Downloads\phase0\Unity\BendViewer"
$manifest = @'
{
  "dependencies": {
    "org.nesnausk.gaussian-splatting": "https://github.com/aras-p/UnityGaussianSplatting.git?path=package",
    "com.unity.render-pipelines.universal": "17.3.0",
    "com.unity.modules.jsonserialize": "1.0.0"
  }
}
'@
# 必須用 UTF8Encoding($false) 確保無 BOM，否則 Unity 報 "Non-whitespace before {" 錯誤
[System.IO.File]::WriteAllText(
    "$PROJ\Packages\manifest.json",
    $manifest,
    (New-Object System.Text.UTF8Encoding $false)
)
```

> **易錯點：**
> - 鍵名必須是 `org.nesnausk.gaussian-splatting`，**不是** `com.aras-p.gaussian-splatting`（repo 的 README 寫錯了，實際 package.json 用 org.nesnausk）
> - 不可用 `Out-File` 或 `ConvertTo-Json`（產生 BOM 或多餘空白，Unity 拒絕解析）

### 步驟二：Batch 匯入 .ply → GaussianSplatAsset

```powershell
$UNITY = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe"
$PROJ  = "C:\Users\User\Downloads\phase0\Unity\BendViewer"
$LOG   = "C:\3d-recon-pipeline\unity_setup\import_log.txt"

# 複製 .ply 到 Assets
Copy-Item "C:\3d-recon-pipeline\exports\3dgs\ply\point_cloud_final.ply" `
          "$PROJ\Assets\GaussianSplats\" -Force

# 執行 batch 匯入
$proc = Start-Process $UNITY -ArgumentList `
    "-batchmode","-nographics",`
    "-projectPath",$PROJ,`
    "-executeMethod","BatchGaussianImport.Run",`
    "-quit","-logFile",$LOG -PassThru
$proc.WaitForExit(600000)  # 最多等 10 分鐘
$proc.ExitCode              # 0 = 成功
```

**結果：** `Assets/GaussianAssets/` 下產出 6 個文件：
- `point_cloud_final.asset`（主資產）
- `*_pos.bytes`（8 MB）、`*_col.bytes`（8 MB）、`*_shs.bytes`（64 MB）、`*_oth.bytes`（16 MB）、`*_chk.bytes`（0.5 MB）

### 步驟三：Batch 建立場景

```powershell
$LOG = "C:\3d-recon-pipeline\unity_setup\create_scene_log.txt"

$proc = Start-Process $UNITY -ArgumentList `
    "-batchmode","-nographics",`
    "-projectPath",$PROJ,`
    "-executeMethod","BatchCreateScene.Run",`
    "-quit","-logFile",$LOG -PassThru
$proc.WaitForExit(300000)  # 最多等 5 分鐘
$proc.ExitCode              # 0 = 成功

# 確認場景存在
Test-Path "$PROJ\Assets\Scenes\FactoryGaussian.unity"  # True
```

**結果：** `Assets/Scenes/FactoryGaussian.unity`（含 Main Camera + GaussianSplatRenderer，m_Asset 指向 point_cloud_final.asset）

### ⚠️ 已知陷阱

| 問題 | 原因 | 修正 |
|------|------|------|
| `error CS1061: 'SplatCount' not found` | 屬性名稱大小寫錯誤 | 應用 `splatCount`（小寫 s） |
| `MissingReferenceException: GaussianSplatAsset has been destroyed` | `EditorSceneManager.NewScene()` 在 batch 模式下會 GC 已載入的 ScriptableObject | 在 `NewScene()` 呼叫後立即重新 `AssetDatabase.LoadAssetAtPath<>()` |
| Compile error: `UniversalAdditionalCameraData` not found | batch 模式下 URP namespace 衝突 | 移除 `using UnityEngine.Rendering.Universal` 及相關程式碼 |
| manifest.json "Non-whitespace before \{" | 有 BOM 或縮排格式問題 | 改用 `[System.IO.File]::WriteAllText` + `UTF8Encoding($false)` |
| package name mismatch | README 寫 `com.aras-p`，實際是 `org.nesnausk` | manifest 鍵名改用 `org.nesnausk.gaussian-splatting` |

### GUI 驗證（Batch 完成後）

```powershell
# 開啟 Unity GUI 驗證
Start-Process "C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe" `
    -ArgumentList "-projectPath","C:\Users\User\Downloads\phase0\Unity\BendViewer"
```

1. Project 面板 → `Assets/Scenes/FactoryGaussian.unity` → 雙擊開啟
2. Hierarchy 確認 `GaussianSplat` GameObject → Inspector → Asset 欄位顯示 `point_cloud_final`
3. 按 Play 或 Scene 視圖中即可見高斯點雲渲染（2,095,607 Gaussians）
