# src/train_3dgs.py
# -*- coding: utf-8 -*-
import sys
import io
import os
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, 
    encoding='utf-8', 
    errors='replace'
)

"""
Phase 1B: 3D Gaussian Splatting Training
========================================
gsplat (Berkeley) from COLMAP sparse/0.

Quick start:
  python -m src.train_3dgs

Full params:
  python -m src.train_3dgs \\
      --imgdir data/frames_1600 \\
      --colmap outputs/SfM_models/sift/sparse/0 \\
      --outdir outputs/3DGS_models \\
      --iterations 30000 \\
      --sh-degree 3
"""

import subprocess, sys, json, shutil
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from src.utils.agent_contracts import find_latest_step_file, infer_outputs_root, write_stage_contract

app   = typer.Typer(help="Phase 1B: gsplat 3DGS 訓練")
console = Console()

DEFAULT_TRAIN_PARAMS = {
    "train_mode": "default",
    "imgdir": "data/frames_1600",
    "colmap": "outputs/SfM_models/sift/sparse/0",
    "outdir": "outputs/3DGS_models",
    "iterations": 30000,
    "sh_degree": 3,
    "densify_until": 15000,
    "scene_scale": 0.0,
    "scale_json": "",
    "eval_steps": 1000,
    "data_factor": 1,
    "absgrad": False,
    "grow_grad2d": 0.0002,
    "antialiased": False,
    "random_bkgd": False,
    "cap_max": 1_000_000,
    "mcmc_min_opacity": None,
    "mcmc_noise_lr": None,
    "opacity_reg": 0.0,
    "pose_opt": False,
    "app_opt": False,
    "disable_video": False,
    "loss_mask_dir": "",
}

MCMC_PRESET_DEFAULTS = {
    "opacity_reg": 0.01,
    "scale_reg": 0.01,
    "min_opacity": 0.005,
    "noise_lr": 500000.0,
    "cap_max": 1_000_000,
}

def _read_json_robust(path: Path) -> dict:
    encodings = ("utf-8", "utf-8-sig", "cp950", "mbcs")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise last_error if last_error is not None else ValueError(f"無法解析 JSON：{path}")


def _check_gsplat() -> bool:
    try:
        import gsplat  # noqa: F401
        return True
    except ImportError:
        return False


def _run(cmd: list[str], cwd: str | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    stale_arch = env.get("TORCH_CUDA_ARCH_LIST")
    if stale_arch:
        console.print(
            f"[yellow]偵測到 TORCH_CUDA_ARCH_LIST={stale_arch}，"
            "目前主線會先移除這個舊設定，再啟動 3DGS trainer。[/]"
        )
        env.pop("TORCH_CUDA_ARCH_LIST", None)

    console.print(">> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _check_pointcloud_validation(validation_report_path: str | None = None) -> bool:
    """
    檢查點雲驗證報告。
    
    Args:
        validation_report_path: 驗證報告路徑（如果 None，使用默認位置）
    
    Returns:
        True 表示可以繼續訓練，False 表示應停止
    """
    # 確定報告文件位置
    if validation_report_path is None:
        project_root = Path(__file__).parent.parent
        validation_report_path = str(project_root / "outputs" / "reports" / "pointcloud_validation_report.json")
    
    report_p = Path(validation_report_path)
    
    # 如果報告不存在，中止（確保 SfM 成功才能訓練）
    if not report_p.exists():
        console.print(
            Panel(
                "[red]❌ 點雲驗證報告不存在（SfM 未成功？）[/]\n\n"
                f"期望位置：{report_p.resolve()}\n\n"
                "解決方案：\n"
                "  1. 先執行 Phase 1A: python -m src.sfm_colmap\n"
                "  2. 確認 SfM 成功生成了報告\n"
                "  3. 再執行 Phase 1B: python -m src.train_3dgs",
                title="[bold red]前置條件未滿足[/]",
                border_style="red",
            )
        )
        raise FileNotFoundError(f"Point cloud validation report not found: {report_p.resolve()}")
    
    # 讀取報告
    try:
        report = _read_json_robust(report_p)
    except Exception as e:
        console.print(
            Panel(
                f"[red]無法解析驗證報告[/]\n{str(e)}",
                title="[bold red]報告錯誤[/]",
                border_style="red",
            )
        )
        raise typer.Exit(1)  # 出錯時停止執行（防止錯誤傳播）
    
    # 檢查是否可以繼續
    can_proceed = report.get("can_proceed_to_3dgs", False)
    
    if not can_proceed:
        diagnosis = report.get("diagnosis", "未知原因")
        console.print(
            Panel(
                f"[red]❌ 點雲品質驗證失敗[/]\n\n"
                f"診斷：{diagnosis}\n\n"
                f"詳情請見：{report_p.resolve()}",
                title="[bold red]無法繼續訓練[/]",
                border_style="red",
            )
        )
        raise typer.Exit(1)
    
    # 成功通過驗證
    console.print(
        Panel(
            "[green]✓ 點雲品質驗證通過[/]",
            title="[bold green]準備訓練[/]",
            border_style="green",
        )
    )
    return True


def _resolve_imgdir(project_root: Path, imgdir: str) -> Path:
    """Prefer the downscaled image set, but fall back to cleaned frames for compatibility."""
    requested = Path(imgdir)
    if not requested.is_absolute():
        requested = project_root / requested

    if requested.exists() and any(requested.iterdir()):
        return requested

    fallback = project_root / "data" / "frames_cleaned"
    if imgdir == "data/frames_1600" and fallback.exists() and any(fallback.iterdir()):
        console.print(
            "[yellow]找不到 data/frames_1600，暫時回退到 data/frames_cleaned。"
            " 若要維持與 SfM 完全一致，建議先重新生成 data/frames_1600。[/]"
        )
        return fallback

    return requested


def _infer_reports_root(project_root: Path, path_value: str) -> Path | None:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = project_root / candidate

    for node in [candidate] + list(candidate.parents):
        if node.name in {"SfM_models", "3DGS_models"}:
            return node.parent / "reports"
    return None


def _resolve_validation_report(
    project_root: Path,
    colmap: str,
    outdir: str,
    validation_report: str,
) -> Path:
    if validation_report:
        path = Path(validation_report)
        return path if path.is_absolute() else project_root / path

    for probe in (colmap, outdir):
        reports_root = _infer_reports_root(project_root, probe)
        if reports_root is not None:
            return reports_root / "pointcloud_validation_report.json"

    return project_root / "outputs" / "reports" / "pointcloud_validation_report.json"


def _resolve_sparse_model_dir(
    project_root: Path,
    colmap: str,
    validation_report_path: Path,
) -> Path:
    requested = Path(colmap)
    if not requested.is_absolute():
        requested = project_root / requested

    if validation_report_path.exists():
        try:
            report = _read_json_robust(validation_report_path)
            sparse_dir = report.get("sparse_dir")
            if sparse_dir:
                report_path = Path(sparse_dir)
                if not report_path.is_absolute():
                    report_path = project_root / report_path
                if report_path.exists():
                    if report_path.resolve() != requested.resolve():
                        console.print(
                            f"[cyan]依驗證報告改用最佳 sparse model[/] {report_path.resolve()}"
                        )
                    return report_path
        except Exception:
            pass

    if requested.is_dir() and not (requested / "points3D.bin").exists():
        sparse_root = requested / "sparse" if (requested / "sparse").exists() else requested
        candidates = []
        if sparse_root.exists():
            for child in sparse_root.iterdir():
                if not child.is_dir():
                    continue
                required = [child / "cameras.bin", child / "images.bin", child / "points3D.bin"]
                if not all(path.exists() and path.stat().st_size > 0 for path in required):
                    continue
                images_size = (child / "images.bin").stat().st_size
                points_size = (child / "points3D.bin").stat().st_size
                candidates.append((child, images_size, points_size))
        if candidates:
            candidates.sort(key=lambda item: (item[1], item[2], item[0].name), reverse=True)
            chosen = candidates[0][0]
            console.print(f"[cyan]自動挑選 sparse model[/] {chosen.resolve()}")
            return chosen

    return requested


def _remove_path(path: Path) -> None:
    """Remove files, normal directories, symlinks, or Windows junctions safely."""
    if not path.exists() and not path.is_symlink():
        return

    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
            return
    except OSError:
        pass

    try:
        os.rmdir(path)
        return
    except OSError:
        shutil.rmtree(path)


def _create_directory_link(link_path: Path, target_path: Path) -> None:
    """Create a directory symlink, or fall back to a Windows junction."""
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
    except OSError:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
            check=True,
        )


def _ensure_scene_dir(scene_dir: Path, img_target: Path, sparse_target: Path) -> None:
    """Build a per-run colmap scene directory to avoid cross-run contamination."""
    images_link = scene_dir / "images"
    sparse_link = scene_dir / "sparse" / "0"

    if scene_dir.exists():
        _remove_path(scene_dir)

    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "sparse").mkdir(parents=True, exist_ok=True)

    _create_directory_link(images_link, img_target)
    _create_directory_link(sparse_link, sparse_target)


def _load_train_params(params_json: str) -> tuple[dict, dict]:
    """Load agent-provided training params.

    Supports either:
      1. a train_params.json file whose top-level already is the plan
      2. a larger payload that contains a "train_params" section
    """
    params_path = Path(params_json)
    if not params_path.exists():
        raise typer.BadParameter(f"找不到 params json：{params_path}")

    payload = _read_json_robust(params_path)
    plan = payload.get("train_params", payload)
    recommended = plan.get("recommended_params", {})
    if not isinstance(recommended, dict):
        raise typer.BadParameter(f"params json 格式不正確：{params_path}")
    return plan, recommended


def _build_eval_schedule(step_interval: int, max_steps: int) -> list[int]:
    """Convert an interval-style setting to the explicit step list expected by simple_trainer."""
    if step_interval <= 0:
        return [max_steps]

    schedule = list(range(step_interval, max_steps + 1, step_interval))
    if max_steps not in schedule:
        schedule.append(max_steps)
    return sorted(set(schedule))


def _resolve_effective_train_config(
    train_mode: str,
    *,
    absgrad: bool,
    grow_grad2d: float,
    antialiased: bool,
    random_bkgd: bool,
    cap_max: int,
    mcmc_min_opacity: float | None,
    mcmc_noise_lr: float | None,
    opacity_reg: float,
    pose_opt: bool,
    app_opt: bool,
) -> dict:
    """Resolve the effective trainer values after preset defaults are applied.

    The wrapper defaults are not always the runtime truth. In particular,
    `mcmc` inherits preset values from gsplat's trainer even when the wrapper
    does not emit an explicit CLI override.
    """
    effective = {
        "absgrad": absgrad,
        "grow_grad2d": grow_grad2d,
        "antialiased": antialiased,
        "random_bkgd": random_bkgd,
        "cap_max": cap_max,
        "mcmc_min_opacity": mcmc_min_opacity,
        "mcmc_noise_lr": mcmc_noise_lr,
        "opacity_reg": opacity_reg,
        "pose_opt": pose_opt,
        "app_opt": app_opt,
        "mcmc_scale_reg": None,
    }

    if train_mode == "mcmc":
        if opacity_reg <= 0.0:
            effective["opacity_reg"] = MCMC_PRESET_DEFAULTS["opacity_reg"]
        effective["mcmc_scale_reg"] = MCMC_PRESET_DEFAULTS["scale_reg"]
        if mcmc_min_opacity is None:
            effective["mcmc_min_opacity"] = MCMC_PRESET_DEFAULTS["min_opacity"]
        if mcmc_noise_lr is None:
            effective["mcmc_noise_lr"] = MCMC_PRESET_DEFAULTS["noise_lr"]
        if cap_max == DEFAULT_TRAIN_PARAMS["cap_max"]:
            effective["cap_max"] = MCMC_PRESET_DEFAULTS["cap_max"]

    return effective


@app.command()
def main(
    train_mode:   str   = typer.Option("default",              help="trainer preset：default 或 mcmc"),
    imgdir:       str   = typer.Option("data/frames_1600",      help="訓練影格目錄（預設優先使用縮圖後的 frames_1600）"),
    colmap:       str   = typer.Option("outputs/SfM_models/sift/sparse/0",   help="COLMAP sparse/0 目錄（相對於專案根目錄）"),
    outdir:       str   = typer.Option("outputs/3DGS_models",         help="輸出目錄（相對於專案根目錄）"),
    iterations:   int   = typer.Option(30000,                  help="訓練迭代數（說明書建議 30000）"),
    sh_degree:    int   = typer.Option(3,                      help="球諧函數階數（金屬場景建議 3）"),
    densify_until:int   = typer.Option(15000,                  help="點雲密化截止迭代"),
    scene_scale:  float = typer.Option(0.0,                    help="scene scale（公尺/unit），0=自動。用 scale_calibrate.py 算出後填入"),
    scale_json:   str   = typer.Option("",                     help="直接讀 scale_calibrate.py 輸出的 JSON 路徑"),
    eval_steps:   int   = typer.Option(1000,                   help="每幾步評估一次 PSNR"),
    data_factor:  int   = typer.Option(1,                      help="影像縮放比例（已縮過就設 1）"),
    absgrad:      bool  = typer.Option(False,                  help="使用 absolute gradients 觸發 densification（AbsGS 方向）"),
    grow_grad2d:  float = typer.Option(0.0002,                 help="2D gradient densification 門檻；配合 absgrad 時常需提高"),
    antialiased:  bool  = typer.Option(False,                  help="開啟 antialiasing rasterization（偏視覺改善）"),
    random_bkgd:  bool  = typer.Option(False,                  help="訓練時使用隨機背景，抑制透明/浮點傾向"),
    cap_max:      int   = typer.Option(1_000_000,              help="MCMC 最大 Gaussian 顆數上限；default preset 會忽略"),
    mcmc_min_opacity: float | None = typer.Option(None,        help="MCMCStrategy 的 min_opacity；未提供時使用 gsplat preset 預設值"),
    mcmc_noise_lr: float | None = typer.Option(None,           help="MCMCStrategy 的 noise_lr；未提供時使用 gsplat preset 預設值"),
    opacity_reg:  float = typer.Option(0.0,                    help="透明度正則化；抑制半透明浮點/浮球"),
    pose_opt:     bool  = typer.Option(False,                  help="啟用 camera pose optimization，微調相機位姿"),
    app_opt:      bool  = typer.Option(False,                  help="啟用 appearance optimization，吸收幀間外觀差異"),
    disable_video: bool = typer.Option(False,                  help="停用 trajectory 影片輸出，適合短訓練 smoke test"),
    loss_mask_dir: str = typer.Option("",                      help="只在訓練 loss 中使用的 mask 目錄；非零像素視為排除區"),
    validation_report: str = typer.Option("",                  help="指定本輪 SfM 驗證報告；未提供時會依 colmap/outdir 自動推導"),
    params_json:  str   = typer.Option("",                     help="由 agent 產生的 train_params.json；提供後會套用建議參數"),
):
    # ── 初期化專案路徑（一次定義） ────────────────────
    project_root = Path(__file__).parent.parent
    
    if params_json:
        plan, recommended = _load_train_params(params_json)
        console.print(
            f"[cyan]套用 agent 訓練設定[/] {Path(params_json).resolve()} "
            f"(profile={plan.get('profile_name', 'unknown')})"
        )
        if train_mode == DEFAULT_TRAIN_PARAMS["train_mode"]:
            train_mode = str(recommended.get("train_mode", train_mode))
        if imgdir == DEFAULT_TRAIN_PARAMS["imgdir"]:
            imgdir = str(recommended.get("imgdir", imgdir))
        if colmap == DEFAULT_TRAIN_PARAMS["colmap"]:
            colmap = str(recommended.get("colmap", colmap))
        if outdir == DEFAULT_TRAIN_PARAMS["outdir"]:
            outdir = str(recommended.get("outdir", outdir))
        if iterations == DEFAULT_TRAIN_PARAMS["iterations"]:
            iterations = int(recommended.get("iterations", iterations))
        if sh_degree == DEFAULT_TRAIN_PARAMS["sh_degree"]:
            sh_degree = int(recommended.get("sh_degree", sh_degree))
        if densify_until == DEFAULT_TRAIN_PARAMS["densify_until"]:
            densify_until = int(recommended.get("densify_until", densify_until))
        if scene_scale == DEFAULT_TRAIN_PARAMS["scene_scale"]:
            scene_scale = float(recommended.get("scene_scale", scene_scale))
        if scale_json == DEFAULT_TRAIN_PARAMS["scale_json"]:
            scale_json = str(recommended.get("scale_json", scale_json))
        if eval_steps == DEFAULT_TRAIN_PARAMS["eval_steps"]:
            eval_steps = int(recommended.get("eval_steps", eval_steps))
        if data_factor == DEFAULT_TRAIN_PARAMS["data_factor"]:
            data_factor = int(recommended.get("data_factor", data_factor))
        if absgrad == DEFAULT_TRAIN_PARAMS["absgrad"]:
            absgrad = bool(recommended.get("absgrad", absgrad))
        if grow_grad2d == DEFAULT_TRAIN_PARAMS["grow_grad2d"]:
            grow_grad2d = float(recommended.get("grow_grad2d", grow_grad2d))
        if antialiased == DEFAULT_TRAIN_PARAMS["antialiased"]:
            antialiased = bool(recommended.get("antialiased", antialiased))
        if random_bkgd == DEFAULT_TRAIN_PARAMS["random_bkgd"]:
            random_bkgd = bool(recommended.get("random_bkgd", random_bkgd))
        if cap_max == DEFAULT_TRAIN_PARAMS["cap_max"]:
            cap_max = int(recommended.get("cap_max", cap_max))
        if mcmc_min_opacity == DEFAULT_TRAIN_PARAMS["mcmc_min_opacity"]:
            mcmc_min_opacity = recommended.get("mcmc_min_opacity", mcmc_min_opacity)
        if mcmc_noise_lr == DEFAULT_TRAIN_PARAMS["mcmc_noise_lr"]:
            mcmc_noise_lr = recommended.get("mcmc_noise_lr", mcmc_noise_lr)
        if opacity_reg == DEFAULT_TRAIN_PARAMS["opacity_reg"]:
            opacity_reg = float(recommended.get("opacity_reg", opacity_reg))
        if pose_opt == DEFAULT_TRAIN_PARAMS["pose_opt"]:
            pose_opt = bool(recommended.get("pose_opt", pose_opt))
        if app_opt == DEFAULT_TRAIN_PARAMS["app_opt"]:
            app_opt = bool(recommended.get("app_opt", app_opt))
        if disable_video == DEFAULT_TRAIN_PARAMS["disable_video"]:
            disable_video = bool(recommended.get("disable_video", disable_video))
        if loss_mask_dir == DEFAULT_TRAIN_PARAMS["loss_mask_dir"]:
            loss_mask_dir = str(recommended.get("loss_mask_dir", loss_mask_dir))

    # ── 前置檢查 ──────────────────────────────────
    if not _check_gsplat():
        console.print(Panel(
            "[red]❌ 找不到可用的 gsplat 環境。[/]\n\n"
            "請使用目前生產層的正式基線，而不是舊的臨時 workaround：\n\n"
            "[yellow]1. 重建乾淨 .venv[/]\n"
            "  Remove-Item .venv -Recurse -Force\n"
            "  python -m venv .venv\n\n"
            "[yellow]2. 依 requirements.txt 重裝[/]\n"
            "  .\\.venv\\Scripts\\Activate.ps1\n"
            "  python -m pip install --upgrade pip setuptools wheel\n"
            "  python -m pip install -r requirements.txt\n\n"
            "[yellow]3. 跑正式 smoke test[/]\n"
            "  python scripts/test_cuda.py\n\n"
            "[yellow]4. 避免舊環境變數[/]\n"
            "  不要再沿用舊的 TORCH_CUDA_ARCH_LIST=12.0\n\n"
            "如果 smoke test 仍失敗，請回頭看：\n"
            "  docs/安裝與環境建置.md\n"
            "  docs/故障排查與急診室.md\n"
            "  文件導航.md（先導航）",
            title="[bold red]缺少相依套件[/]",
            border_style="red",
        ))
        raise typer.Exit(1)

    train_mode = train_mode.strip().lower()
    if train_mode not in {"default", "mcmc"}:
        console.print(f"[red]不支援的 train_mode：{train_mode}（只允許 default / mcmc）[/]")
        raise typer.Exit(1)
    
    # ── 檢查點雲驗證報告 ────────────────────────────
    validation_report_path = _resolve_validation_report(
        project_root=project_root,
        colmap=colmap,
        outdir=outdir,
        validation_report=validation_report,
    )
    _check_pointcloud_validation(str(validation_report_path))

    img_p    = _resolve_imgdir(project_root, imgdir)
    colmap_p = _resolve_sparse_model_dir(project_root, colmap, validation_report_path)
    out_p    = project_root / outdir  # 绝对路径：project_root/outputs/3DGS_models

    if not img_p.exists() or not any(img_p.iterdir()):
        console.print(f"[red]影格目錄為空：{img_p.resolve()}[/]")
        raise typer.Exit(1)
    if not colmap_p.exists():
        console.print(f"[red]COLMAP 目錄不存在：{colmap_p.resolve()}[/]")
        raise typer.Exit(1)
    loss_mask_p = None
    if loss_mask_dir:
        loss_mask_p = Path(loss_mask_dir)
        if not loss_mask_p.is_absolute():
            loss_mask_p = project_root / loss_mask_p
        if not loss_mask_p.exists():
            console.print(f"[red]loss mask 目錄不存在：{loss_mask_p.resolve()}[/]")
            raise typer.Exit(1)

    n_imgs = len(list(img_p.glob("*.png"))) + len(list(img_p.glob("*.jpg")))
    console.print(f"[cyan]影格[/]    {img_p}  ({n_imgs} 張)")
    console.print(f"[cyan]COLMAP[/]  {colmap_p}")
    console.print(f"[cyan]輸出[/]    {out_p}")

    out_p.mkdir(parents=True, exist_ok=True)

    # ── 讀取 scale（可選）────────────────────────
    actual_scale = scene_scale
    if scale_json:
        sj = Path(scale_json)
        if sj.exists():
            data = json.loads(sj.read_text())
            actual_scale = data.get("scale_m_per_unit", 0.0)
            console.print(f"[green]scale_factor = {actual_scale:.6f} m/unit（來自 {sj.name}）[/]")
        else:
            console.print(f"[yellow]找不到 {sj}，使用預設 scale[/]")
    elif actual_scale == 0.0:
        console.print("[yellow]scene_scale = 0（自動估計）。若要精確公尺單位，請先跑 scale_calibrate.py[/]")

    # Build a per-run scene dir so different experiments do not share stale junctions.
    trainer      = project_root / "gsplat_runner" / "simple_trainer.py"
    scene_dir    = out_p / "_colmap_scene"

    if not trainer.exists():
        console.print(f"[red]Cannot find {trainer}[/]")
        raise typer.Exit(1)

    img_target = img_p.resolve()
    sparse_target = colmap_p.resolve()
    console.print("[yellow]Building isolated colmap scene directory for this run...[/]")
    _ensure_scene_dir(scene_dir, img_target, sparse_target)
    console.print(f"  images -> {img_target}")
    console.print(f"  sparse/0 -> {sparse_target}")

    # simple_trainer args
    eval_schedule = _build_eval_schedule(eval_steps, iterations)

    base_args = [
        train_mode,
        "--data-dir",    str(scene_dir),
        "--result-dir",  str(out_p.resolve()),
        "--max-steps",   str(iterations),
        "--sh-degree",   str(sh_degree),
        "--data-factor", str(data_factor),
        "--disable-viewer",
        "--init-type",   "sfm",
        "--strategy.refine-stop-iter", str(densify_until),
    ]

    if train_mode == "default":
        base_args.extend(["--strategy.grow-grad2d", str(grow_grad2d)])
    elif train_mode == "mcmc":
        base_args.extend(["--strategy.cap-max", str(cap_max)])
        if mcmc_min_opacity is not None:
            base_args.extend(["--strategy.min-opacity", str(mcmc_min_opacity)])
        if mcmc_noise_lr is not None:
            base_args.extend(["--strategy.noise-lr", str(mcmc_noise_lr)])

    if eval_schedule:
        base_args.append("--eval-steps")
        base_args.extend(str(step) for step in eval_schedule)

    if train_mode == "default" and absgrad:
        base_args.append("--strategy.absgrad")
    if antialiased:
        base_args.append("--antialiased")
    if random_bkgd:
        base_args.append("--random-bkgd")
    if opacity_reg > 0.0:
        base_args.extend(["--opacity-reg", str(opacity_reg)])
    if pose_opt:
        base_args.append("--pose-opt")
    if app_opt:
        base_args.append("--app-opt")
    if disable_video:
        base_args.append("--disable-video")
    if loss_mask_p is not None:
        base_args.extend(["--loss-mask-dir", str(loss_mask_p.resolve())])
    
    # 加入 scene_scale（如果有指定）
    if actual_scale != 0.0:
        base_args.extend(["--global-scale", f"{actual_scale:.6f}"])

    cmd = [sys.executable, str(trainer)] + base_args
    runner_dir = str(trainer.parent)  # cwd = gsplat_runner/
    effective_cfg = _resolve_effective_train_config(
        train_mode,
        absgrad=absgrad,
        grow_grad2d=grow_grad2d,
        antialiased=antialiased,
        random_bkgd=random_bkgd,
        cap_max=cap_max,
        mcmc_min_opacity=mcmc_min_opacity,
        mcmc_noise_lr=mcmc_noise_lr,
        opacity_reg=opacity_reg,
        pose_opt=pose_opt,
        app_opt=app_opt,
    )

    console.print(f"[dim]cwd: {runner_dir}[/]")

    console.print()
    summary_lines = [
        f"Preset:      {train_mode}",
        f"迭代數:     {iterations:,}",
        f"球諧階數:   {sh_degree}",
        f"密化截止:   {densify_until:,}",
        f"AbsGrad:    {'ON' if effective_cfg['absgrad'] else 'OFF'}",
        f"grow_grad2d:{effective_cfg['grow_grad2d']:.4f}",
        f"AA:         {'ON' if effective_cfg['antialiased'] else 'OFF'}",
        f"Rnd Bkgd:   {'ON' if effective_cfg['random_bkgd'] else 'OFF'}",
        f"Cap Max:    {effective_cfg['cap_max']:,}",
        f"OpacityReg: {effective_cfg['opacity_reg']:.4f}",
        f"Pose Opt:   {'ON' if effective_cfg['pose_opt'] else 'OFF'}",
        f"App Opt:    {'ON' if effective_cfg['app_opt'] else 'OFF'}",
    ]
    if train_mode == "mcmc":
        summary_lines.extend(
            [
                f"MCMC min_opa: {effective_cfg['mcmc_min_opacity']:.4f}",
                f"MCMC noise_lr:{effective_cfg['mcmc_noise_lr']:.0f}",
                f"MCMC scale_reg:{effective_cfg['mcmc_scale_reg']:.4f}",
            ]
        )
    summary_lines.extend(
        [
            f"Video:      {'OFF' if disable_video else 'ON'}",
            f"Loss Mask:  {str(loss_mask_p.resolve()) if loss_mask_p is not None else 'OFF'}",
            f"評估步點:   {', '.join(str(step) for step in eval_schedule[:5])}"
            + (" ..." if len(eval_schedule) > 5 else ""),
            f"Scene scale: {'自動' if actual_scale == 0 else f'{actual_scale:.6f} m/unit'}",
            "",
            "[dim]註：此處顯示的是 trainer 最終生效值，不是 wrapper 預設值。[/]",
            "[dim]RTX 5070 Ti 預計訓練時間：~60-90 min[/]",
        ]
    )
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="[bold green]Phase 1B Training Params[/]",
            border_style="green",
        )
    )

    try:
        _run(cmd, cwd=runner_dir)
    except subprocess.CalledProcessError:
        console.print(Panel(
            "訓練中途失敗。常見原因：\n"
            "  1. GPU 顯存不足 → 減少 --densify-until 或降低影像解析度\n"
            "  2. gsplat CUDA kernel 未編譯 → 嘗試 pip install gsplat --no-cache-dir\n"
            "  3. simple_trainer 參數版本不符 → pip install --upgrade gsplat",
            title="[bold red]訓練失敗[/]",
            border_style="red",
        ))
        raise typer.Exit(1)

    # ── 完成後輸出摘要 ────────────────────────────
    splats = list(out_p.rglob("*.ply")) + list(out_p.rglob("*.splat"))
    console.print(Panel(
        f"輸出目錄：{out_p.resolve()}\n"
        + ("\n".join(f"  {p.relative_to(out_p)}" for p in splats) if splats else "  （尚無 .ply/.splat，請確認輸出格式）"),
        title="[bold green]Phase 1B 完成 ✓[/]",
        border_style="green",
    ))
    console.print("\n[bold]Phase 2 Next:[/]")
    console.print("  -> python -m src.export_ply")
    console.print("  -> python -m src.export_ply_unity --unity")

    latest_stats = find_latest_step_file(out_p, "val_step*.json", "val_step")
    latest_ckpt = find_latest_step_file(out_p, "ckpt_*_rank0.pt", "ckpt_")
    metrics: dict[str, float | int | None] = {
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "num_gs": None,
    }
    if latest_stats is not None and latest_stats.exists():
        try:
            stats_payload = _read_json_robust(latest_stats)
            metrics["psnr"] = stats_payload.get("psnr")
            metrics["ssim"] = stats_payload.get("ssim")
            metrics["lpips"] = stats_payload.get("lpips")
            metrics["num_gs"] = stats_payload.get("num_GS", stats_payload.get("num_gs"))
        except Exception:
            pass

    outputs_root = infer_outputs_root(project_root, out_p)
    contract_paths = write_stage_contract(
        project_root=project_root,
        run_root=outputs_root,
        stage="train_complete",
        status="completed",
        artifacts={
            "pointcloud_report": validation_report_path,
            "colmap_dir": colmap_p,
            "result_dir": out_p,
            "stats_json": latest_stats,
            "checkpoint": latest_ckpt,
        },
        metrics=metrics,
        params={
            "train_mode": train_mode,
            "imgdir": str(img_p.resolve()),
            "iterations": iterations,
            "sh_degree": sh_degree,
            "densify_until": densify_until,
            "eval_steps": eval_steps,
            "data_factor": data_factor,
            "absgrad": absgrad,
            "grow_grad2d": grow_grad2d,
            "antialiased": antialiased,
            "random_bkgd": random_bkgd,
            "cap_max": cap_max,
            "opacity_reg": opacity_reg,
            "pose_opt": pose_opt,
            "app_opt": app_opt,
            "loss_mask_dir": str(loss_mask_p.resolve()) if loss_mask_p is not None else "",
        },
        summary=f"3DGS {train_mode} training complete",
    )
    console.print(f"[green]Agent contract 已導出[/] {contract_paths['local_contract']}")
    console.print(f"[green]Agent event 已導出[/] {contract_paths['event_file']}")


if __name__ == "__main__":
    app()
