# Experimental Scripts

這個目錄存放目前不列為正式生產層主線的腳本。

目前收納內容：

- `sfm_colmap_aliked.py`
  - ALIKED 對比支線，尚未成為正式 SfM 主線
- `create_validation_set.py`
  - 尚未補完整驗證閉環的資料集工具
- `diagnose_video.py`
  - 手動診斷輔助工具，不是正式 pipeline 入口

原則：

- `src/` 只保留正式生產層主線腳本
- `experimental/` 保留研究支線與可回收工具
- 若之後完成驗證，可再移回正式主線
