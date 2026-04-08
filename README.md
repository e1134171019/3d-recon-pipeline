# 工業 3D 重建管線

**狀態**: current  
**最後整理**: 2026-04-08  
**正式輸出根目錄**: `outputs/`

Windows 上的工業場景 3D 重建管線：影片/影像前處理 → COLMAP SfM → gsplat 3DGS 訓練 → 匯出 / Unity。

---

## 專案目前在做什麼

這個 repo 的正式主線不是單一演算法庫，而是一條可分階段執行的流程：

1. 從影片抽幀並做品質篩選
2. 用 COLMAP 重建稀疏點雲與相機位姿
3. 用 gsplat 進行 3D Gaussian Splatting 訓練
4. 匯出 PLY 或往 Unity 流程銜接

目前首頁只描述 repo 內已存在且可對上的主線腳本。舊版路徑、研究規劃、外部協作流程不再作為主線說明。

---

## 正式主線流程

### Phase 0: 視訊前處理

- 腳本: `python -m src.preprocess_phase0`
- 主要輸入: `data/viode/`
- 主要輸出: `data/frames_cleaned/`

用途：

- 從原始影片抽出影格
- 基於清晰度與亮度做品質過濾
- 生成後續 SfM 可用的影格集

補充：

- 目前這批工業資料直接用 `data/frames_cleaned/` 跑 SfM 會偏慢
- 正式主線建議在 Phase 0 後先執行 `src.downscale_frames.py`，生成 `data/frames_1600/`
- 後續 Phase 1A 與 Phase 1B 目前都以 `data/frames_1600/` 作為優先主線，避免前後影像集不一致

### Phase 1A: SfM / COLMAP

- 腳本: `python -m src.sfm_colmap`
- 目前預設輸入: `data/frames_1600/`
- 正式輸出: `outputs/SfM_models/sift/`
- 驗證報告: `outputs/reports/pointcloud_validation_report.json`

用途：

- 特徵提取
- 影像匹配
- 稀疏重建
- 產生供 3DGS 訓練使用的 sparse model

### Phase 1B: 3DGS 訓練

- 腳本: `python -m src.train_3dgs`
- 目前預設影像: `data/frames_1600/`
- 預設 COLMAP 輸入: `outputs/SfM_models/sift/sparse/0`
- 正式輸出: `outputs/3DGS_models/`

用途：

- 讀取 COLMAP sparse model
- 建立 gsplat 訓練場景
- 產生 checkpoints、stats、renders、tensorboard 輸出

註：

- `imgdir` 建議與 Phase 1A 使用的影像集保持一致
- 若 `data/frames_1600/` 暫時不存在，`src.train_3dgs.py` 會回退到 `data/frames_cleaned/`
- 目前正式建議仍是先補齊 `data/frames_1600/`，讓 SfM 和 3DGS 使用同一批影像

### Phase 2: 匯出 / Unity

- PLY 匯出: `python -m src.export_ply`
- Unity 相關工具: `unity_setup/`、`src/export_ply_unity.py`

---

## 最短使用路徑

以下命令假設你的 Windows / Python / CUDA / COLMAP 環境已經準備好。

```powershell
cd C:\3d-recon-pipeline

# Phase 0
python -m src.preprocess_phase0

# Phase 0.5（目前推薦主線）
python -m src.downscale_frames --src data/frames_cleaned --dst data/frames_1600 --max-side 1600

# Phase 1A
python -m src.sfm_colmap --imgdir data/frames_1600

# Phase 1B
python -m src.train_3dgs --imgdir data/frames_1600

# Phase 2
python -m src.export_ply --ckpt outputs/3DGS_models/ckpts/ckpt_29999_rank0.pt --out outputs/3DGS_models/ply/point_cloud_final.ply
```

## 正式 Smoke Test

目前只保留一支正式 smoke test：

```powershell
python test_cuda.py
```

其餘歷史排障工具已移除或封存，不再視為主線入口。

### 已驗證的 Windows 3DGS smoke test

2026-04-08 這個工作區已經完成一次可跑通的 Windows 短訓練 smoke test。

- 輸出位置：`outputs/3DGS_smoke_sdk_pass/`
- 已確認產生：
  - `stats/val_step0009.json`
  - `ckpts/ckpt_19_rank0.pt`
  - `videos/traj_9.mp4`

這次驗證代表：

- `gsplat` 的 Windows JIT 編譯在目前機器上可成功完成
- 20 step 短訓練可實際跑完
- `imageio-ffmpeg` 已是正式 runtime 依賴之一

本機驗證時，3DGS 需要在帶有 Visual Studio + Windows SDK / UCRT 的終端中啟動；若只開一般 shell，仍可能遇到 `corecrt.h` 之類的工具鏈錯誤。

### 已驗證的正式 30k 訓練與匯出

同樣在 2026-04-08，這個工作區也已完成一次正式 `30,000` step 訓練與 PLY 匯出。

正式產物位置：

- checkpoint：`outputs/3DGS_models/ckpts/ckpt_29999_rank0.pt`
- 驗證指標：`outputs/3DGS_models/stats/val_step0999.json`
- 最終 PLY：`outputs/3DGS_models/ply/point_cloud_final.ply`
- 軌跡影片：`outputs/3DGS_models/videos/traj_999.mp4`

目前已記錄到的驗證指標：

- `PSNR = 18.8670`
- `SSIM = 0.7096`
- `LPIPS = 0.4654`

PLY 匯出已驗證可用；在目前這版環境中，`src.export_ply.py` 會自動回退到手動 PLY 寫出流程，因此不依賴 `gsplat.utils.export_splats` 也可完成匯出。

如果你尚未整理好本機環境，先看：

- [安裝指南.md](/C:/3d-recon-pipeline/安裝指南.md)
- [環境設置.md](/C:/3d-recon-pipeline/環境設置.md)
- [故障排查.md](/C:/3d-recon-pipeline/故障排查.md)
- [文件重整藍圖.md](/C:/3d-recon-pipeline/文件重整藍圖.md)

---

## 主要腳本

| 腳本 | 階段 | 用途 | 目前正式輸出 |
|------|------|------|-------------|
| `src/preprocess_phase0.py` | Phase 0 | 影片抽幀與品質篩選 | `data/frames_cleaned/` |
| `src/downscale_frames.py` | Phase 0.5 | 影像縮圖 | 目標目錄由 `--dst` 指定 |
| `src/sfm_colmap.py` | Phase 1A | COLMAP SfM 主線 | `outputs/SfM_models/sift/` |
| `src/train_3dgs.py` | Phase 1B | gsplat 訓練入口 | `outputs/3DGS_models/` |
| `src/export_ply.py` | Phase 2 | 匯出 PLY | 匯出路徑由參數指定 |
| `src/export_ply_unity.py` | Phase 2 | Unity 相關匯出 | 依參數指定 |

---

## 正式目錄與舊路徑

### 正式路徑

- SfM: `outputs/SfM_models/`
- 報告: `outputs/reports/`
- 3DGS: `outputs/3DGS_models/`

### 舊路徑

- `exports/` 目前視為 `legacy`

如果你看到文件或腳本提到 `exports/3dgs_auto`、`exports/3dgs_output`、`exports/3dgs`，那是舊流程或過渡產物，不是現在首頁主線。

---

## 目前不列為主線的內容

以下內容仍有參考價值，但不應再被視為首頁主流程：

- `D:\agent_test` 外部協作 / orchestration 流程
- `exports/3dgs_auto` 舊版輸出路徑
- `Phase-0A / Phase-0B` 研究型敘事
- `ALIKED` 對比支線
- 各種臨時 Windows / CUDA patch 記錄
- 已封存的歷史排障工具

這些內容後續會逐步移到獨立文件或封存區。

目前已實際移出的研究 / 工具腳本位於：

- `experimental/`

---

## 未來擴展

### ALIKED + LightGlue 對比支線

**狀態**: experimental

定位：

- 作為現行 `SIFT + sequential matcher` 的對比支線
- 用來評估在低紋理、反光、金屬工業場景下的特徵與匹配品質

預期優勢：

- 更強的低紋理場景表現
- 對反光與金屬表面更有韌性

成本：

- 需要額外整合與驗證
- 需要更多實測來確認是否值得升級為正式可選方案

正式評估時應比較以下指標：

- `avg_features_per_image`
- `num_matches`
- `inlier_ratio`
- `registered_images`
- `points3d_count`
- 3DGS 後續 `PSNR / SSIM / LPIPS`
- 總耗時
- GPU 記憶體占用

在沒有完成實測前，README 不會把這條支線寫成已完成能力。

---

## 文件索引

### 現行基準文件

- [文件重整藍圖.md](/C:/3d-recon-pipeline/文件重整藍圖.md)

### 現行說明書

- [安裝指南.md](/C:/3d-recon-pipeline/安裝指南.md)
- [環境設置.md](/C:/3d-recon-pipeline/環境設置.md)
- [故障排查.md](/C:/3d-recon-pipeline/故障排查.md)

### 背景 / 研究文件

- [Phase0_掃描說明書.md](/C:/3d-recon-pipeline/Phase0_掃描說明書.md)

---

## 近期整理方向

接下來的整理順序會是：

1. 重建乾淨 `.venv`
2. 先跑 `python test_cuda.py`
3. 做一次最小 3DGS smoke test
4. 再啟動完整訓練
