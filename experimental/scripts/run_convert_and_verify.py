"""
臨時驗證腳本：執行 Labelme -> YOLO 轉換並輸出驗證報告
"""
import sys
from pathlib import Path

# 確保專案根目錄在 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.convert_labelme_to_yolo_seg import convert_labelme_dir

IMAGES_DIR  = PROJECT_ROOT / "data/annotations/bootstrap_12/images"
LABELS_DIR  = PROJECT_ROOT / "data/annotations/bootstrap_12/labels"
DATASET_YAML = PROJECT_ROOT / "data/annotations/bootstrap_12/dataset.yaml"

# ── Step 1: 執行轉換 ──────────────────────────────────────
print("=" * 60)
print("Step 1: 轉換 Labelme JSON -> YOLO-seg txt")
print("=" * 60)

summary = convert_labelme_dir(
    images_dir=IMAGES_DIR,
    labels_dir=LABELS_DIR,
    dataset_yaml=DATASET_YAML,
    class_name="punch_holders",
    accepted_labels={"punch_holders", "上夾模塊"},
)

print(f"  images     : {summary['images']}")
print(f"  positives  : {summary['positives']}  (有 punch_holders 標注)")
print(f"  negatives  : {summary['negatives']}  (空 txt，負樣本)")
print(f"  kept_shapes: {summary['kept_shapes']}  (總 polygon 數)")

# ── Step 2: 逐檔驗證 ──────────────────────────────────────
print()
print("=" * 60)
print("Step 2: 逐檔驗證")
print("=" * 60)

all_txt = sorted(LABELS_DIR.glob("*.txt"))
issues = []

for txt in all_txt:
    content = txt.read_text(encoding="utf-8").strip()
    lines = [l for l in content.splitlines() if l.strip()]
    
    # 判斷正樣本或負樣本
    if not lines:
        tag = "  [NEG]"
    else:
        tag = f"  [POS] {len(lines)} shape(s)"
        # 驗證格式：每行應以 "0 " 開頭，且有偶數個 float
        for i, line in enumerate(lines):
            parts = line.split()
            if parts[0] != "0":
                issues.append(f"  ⚠  {txt.name} line {i+1}: class_id={parts[0]} (非 0)")
            coord_parts = parts[1:]
            if len(coord_parts) % 2 != 0:
                issues.append(f"  ⚠  {txt.name} line {i+1}: 座標數量不為偶數 ({len(coord_parts)})")
            if len(coord_parts) < 6:
                issues.append(f"  ⚠  {txt.name} line {i+1}: 座標點數 < 3 (只有 {len(coord_parts)//2} 點)")
    
    print(f"  {txt.name}{tag}")

# ── Step 3: 檢查 json 缺失的 2 張 ──────────────────────────
print()
print("=" * 60)
print("Step 3: 確認負樣本（無 json 的圖）")
print("=" * 60)
for frame in ["frame_000337", "frame_000635"]:
    jpg = IMAGES_DIR / f"{frame}.jpg"
    json_ = IMAGES_DIR / f"{frame}.json"
    txt = LABELS_DIR / f"{frame}.txt"
    print(f"  {frame}")
    print(f"    jpg  : {'✓' if jpg.exists() else '✗'}")
    print(f"    json : {'✗ (無，正常)' if not json_.exists() else '⚠ 存在（預期不存在）'}")
    print(f"    txt  : {'✓ 空 (負樣本)' if txt.exists() and not txt.read_text().strip() else '⚠ 非空或不存在'}")

# ── Step 4: 總結 ────────────────────────────────────────
print()
print("=" * 60)
print("Step 4: dataset.yaml 內容")
print("=" * 60)
print(DATASET_YAML.read_text(encoding="utf-8"))

print("=" * 60)
if issues:
    print(f"⚠  發現 {len(issues)} 個問題：")
    for iss in issues:
        print(iss)
else:
    print("✅ 所有驗證通過，無問題。")
print("=" * 60)
