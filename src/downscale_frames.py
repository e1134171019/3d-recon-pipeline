from pathlib import Path
import cv2
from tqdm import tqdm
import typer

app = typer.Typer()

@app.command()
def main(src: str="data/frames", dst: str="data/frames_1k", max_side: int=1000):
    src_p, dst_p = Path(src), Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src_p.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])
    if not files:
        raise SystemExit(f"找不到影格：{src_p}")
    for f in tqdm(files, desc=f"Downscale to <= {max_side}px"):
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_p / f.name), im)
    print(f"[OK] {len(files)} 張 → {dst}")

if __name__ == "__main__":
    app()
