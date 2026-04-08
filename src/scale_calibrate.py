# src/scale_calibrate.py
"""
Scale Calibration Tool
======================
從 COLMAP points3D.bin 計算「1 COLMAP unit = X mm」的比例係數。

兩種模式
--------
1. --mode a4
   指定四個已知角點的 point3D ID（A4 紙診角方案）：
     python -m src.scale_calibrate --mode a4 --ids 10 20 30 40

2. --mode ruler
   指定兩個點的 point3D ID + 它們的真實距離(mm)：
     python -m src.scale_calibrate --mode ruler --ids 10 20 --real-mm 297

3. --mode list
   列出所有點（ID, XYZ, 誤差），方便你挑選角點：
     python -m src.scale_calibrate --mode list --max-err 1.0
"""

import struct
import math
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="COLMAP → 公尺 scale 校正")
console = Console()

# ──────────────────────────────────────────────
# COLMAP binary reader
# ──────────────────────────────────────────────

def read_points3d_bin(path: Path) -> dict:
    """讀 COLMAP points3D.bin，返回 {point3D_id: {'xyz': [x,y,z], 'rgb': [...], 'error': float}}"""
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pid   = struct.unpack("<Q", f.read(8))[0]
            xyz   = struct.unpack("<ddd", f.read(24))
            rgb   = struct.unpack("<BBB", f.read(3))
            err   = struct.unpack("<d", f.read(8))[0]
            n_vis = struct.unpack("<Q", f.read(8))[0]
            f.read(n_vis * 8)           # skip track: image_id(uint32) + point2D_idx(uint32) = 8 bytes each
            points[pid] = {"xyz": list(xyz), "rgb": list(rgb), "error": err}
    return points


def dist3d(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────

@app.command()
def main(
    points_bin: str = "outputs/SfM_models/sift/sparse/0/points3D.bin",
    mode: str = typer.Option("list", help="模式：list | a4 | ruler"),
    ids: list[int] = typer.Option([], help="指定 point3D ID（list 模式不需要）"),
    real_mm: float = typer.Option(0.0, help="ruler 模式：兩點間的真實距離(mm)"),
    max_err: float = typer.Option(2.0, help="list 模式：只顯示重投影誤差 <= 此值的點"),
    top_n: int = typer.Option(50, help="list 模式：顯示前 N 個低誤差點"),
    out: str = typer.Option("outputs/reports/scale_calibration.json", help="輸出 JSON 路徑"),
):
    pb = Path(points_bin)
    if not pb.exists():
        console.print(f"[red][ERR] 找不到 {pb.resolve()}[/]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]讀取[/] {pb} …")
    pts = read_points3d_bin(pb)
    console.print(f"  共 {len(pts):,} 個 3D 點")

    # ── mode: list ──────────────────────────────
    if mode == "list":
        filtered = [(pid, d) for pid, d in pts.items() if d["error"] <= max_err]
        filtered.sort(key=lambda x: x[1]["error"])
        filtered = filtered[:top_n]

        table = Table(title=f"低誤差點（≤ {max_err}px），前 {top_n} 個")
        table.add_column("ID", justify="right")
        table.add_column("X", justify="right")
        table.add_column("Y", justify="right")
        table.add_column("Z", justify="right")
        table.add_column("Error(px)", justify="right")
        for pid, d in filtered:
            x, y, z = d["xyz"]
            table.add_row(
                str(pid),
                f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                f"{d['error']:.3f}",
            )
        console.print(table)
        console.print("\n[yellow]提示：從上表挑選 A4 角點或尺端點的 ID，再用 --mode a4 / --mode ruler 計算 scale。[/]")
        return

    # ── 取得指定點 ───────────────────────────────
    if not ids:
        console.print("[red]請用 --ids 指定點的 ID[/]")
        raise typer.Exit(1)

    selected = {}
    for pid in ids:
        if pid not in pts:
            console.print(f"[red]point3D ID {pid} 不在點雲中[/]")
            raise typer.Exit(1)
        selected[pid] = pts[pid]
        console.print(f"  ID {pid}: xyz={[round(v,4) for v in pts[pid]['xyz']]}, err={pts[pid]['error']:.3f}px")

    # ── mode: ruler ──────────────────────────────
    if mode == "ruler":
        if len(ids) != 2:
            console.print("[red]ruler 模式需要剛好 2 個 ID[/]")
            raise typer.Exit(1)
        if real_mm <= 0:
            console.print("[red]ruler 模式需要 --real-mm（真實距離，mm）[/]")
            raise typer.Exit(1)
        a, b = [pts[i]["xyz"] for i in ids]
        colmap_dist = dist3d(a, b)
        scale = real_mm / colmap_dist
        console.print(f"\n  COLMAP 距離  = {colmap_dist:.6f} units")
        console.print(f"  真實距離     = {real_mm} mm")
        console.print(f"[bold green]  scale_factor = {scale:.6f} mm/unit[/]")
        _save(out, scale, ids, mode)
        return

    # ── mode: a4 ─────────────────────────────────
    if mode == "a4":
        if len(ids) < 2:
            console.print("[red]a4 模式至少需要 2 個 ID（建議 4 個角點）[/]")
            raise typer.Exit(1)

        # A4 對角線 = sqrt(210² + 297²) ≈ 363.8 mm
        A4_DIAG_MM = math.sqrt(210**2 + 297**2)
        coords = [pts[i]["xyz"] for i in ids]

        if len(ids) == 4:
            # 計算兩條對角線的平均
            d1 = dist3d(coords[0], coords[2])
            d2 = dist3d(coords[1], coords[3])
            colmap_diag = (d1 + d2) / 2
            console.print(f"  對角線1 = {d1:.6f}, 對角線2 = {d2:.6f}, 平均 = {colmap_diag:.6f}")
        else:
            # 只有 2 點，用對角線方向距離逼近
            colmap_diag = dist3d(coords[0], coords[1])
            console.print(f"  兩點距離 = {colmap_diag:.6f}（假設為 A4 對角線）")

        scale = A4_DIAG_MM / colmap_diag
        console.print(f"  A4 對角線真實 = {A4_DIAG_MM:.2f} mm")
        console.print(f"[bold green]  scale_factor  = {scale:.6f} mm/unit[/]")
        _save(out, scale, ids, mode)
        return

    console.print(f"[red]未知模式：{mode}，請用 list / a4 / ruler[/]")
    raise typer.Exit(1)


def _save(out_path: str, scale: float, ids: list, mode: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "scale_mm_per_unit": scale,
        "scale_m_per_unit": scale / 1000,
        "mode": mode,
        "reference_ids": ids,
    }
    out.write_text(json.dumps(data, indent=2))
    console.print(f"\n[bold]已存至[/] {out.resolve()}")
    console.print(f"  → train_3dgs.py 使用時加上 --scene-scale {scale/1000:.6f}")


if __name__ == "__main__":
    app()
