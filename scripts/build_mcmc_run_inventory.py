from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "outputs" / "experiments"
REPORTS_ROOT = ROOT / "outputs" / "reports"
TRAIN_PROBES_ROOT = EXPERIMENTS_ROOT / "train_probes"
MARATHON_ROOT = EXPERIMENTS_ROOT / "mcmc_marathon"


@dataclass
class RunRecord:
    source_collection: str
    batch_or_wrapper: str
    run_name: str
    run_dir: str
    cfg_path: str
    metrics_path: str
    completed: bool
    max_steps: int | None
    latest_val_step: int | None
    antialiased: bool | None
    cap_max: int | None
    min_opacity: float | None
    noise_lr: float | None
    refine_every: int | None
    refine_start_iter: int | None
    refine_stop_iter: int | None
    ssim_lambda: float | None
    random_bkgd: bool | None
    scale_reg: float | None
    opacity_reg: float | None
    use_bilateral_grid: bool | None
    depth_loss: bool | None
    with_ut: bool | None
    with_eval3d: bool | None
    visible_adam: bool | None
    pose_opt: bool | None
    app_opt: bool | None
    psnr: float | None
    ssim: float | None
    lpips: float | None
    num_gs: int | None
    unity_log_count: int

    def to_row(self) -> dict[str, Any]:
        return {
            "source_collection": self.source_collection,
            "batch_or_wrapper": self.batch_or_wrapper,
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "cfg_path": self.cfg_path,
            "metrics_path": self.metrics_path,
            "completed": self.completed,
            "max_steps": self.max_steps,
            "latest_val_step": self.latest_val_step,
            "antialiased": self.antialiased,
            "cap_max": self.cap_max,
            "min_opacity": self.min_opacity,
            "noise_lr": self.noise_lr,
            "refine_every": self.refine_every,
            "refine_start_iter": self.refine_start_iter,
            "refine_stop_iter": self.refine_stop_iter,
            "ssim_lambda": self.ssim_lambda,
            "random_bkgd": self.random_bkgd,
            "scale_reg": self.scale_reg,
            "opacity_reg": self.opacity_reg,
            "use_bilateral_grid": self.use_bilateral_grid,
            "depth_loss": self.depth_loss,
            "with_ut": self.with_ut,
            "with_eval3d": self.with_eval3d,
            "visible_adam": self.visible_adam,
            "pose_opt": self.pose_opt,
            "app_opt": self.app_opt,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "num_gs": self.num_gs,
            "unity_log_count": self.unity_log_count,
        }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_yaml(cfg_path: Path) -> dict[str, Any]:
    text = cfg_path.read_text(encoding="utf-8")
    return yaml.unsafe_load(text) or {}


def _load_latest_metrics(stats_dir: Path) -> tuple[Path | None, dict[str, Any], int | None]:
    metric_files = sorted(stats_dir.glob("val_step*.json"))
    if not metric_files:
        return None, {}, None

    def key_func(path: Path) -> int:
        match = re.search(r"val_step(\d+)\.json$", path.name)
        return int(match.group(1)) if match else -1

    latest = max(metric_files, key=key_func)
    step = key_func(latest)
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return latest, payload, step


def _extract_strategy(cfg: dict[str, Any]) -> dict[str, Any]:
    strategy = cfg.get("strategy")
    if isinstance(strategy, dict):
        return strategy
    if hasattr(strategy, "__dict__"):
        payload = getattr(strategy, "__dict__")
        if isinstance(payload, dict):
            return payload
    return {}


def _build_record(source_collection: str, batch_or_wrapper: str, run_dir: Path, cfg_path: Path) -> RunRecord:
    cfg = _read_yaml(cfg_path)
    strategy = _extract_strategy(cfg)
    stats_dir = run_dir / "stats"
    metrics_path, metrics_payload, latest_val_step = _load_latest_metrics(stats_dir)
    max_steps = _safe_int(cfg.get("max_steps"))
    completed = bool(
        max_steps is not None and latest_val_step is not None and latest_val_step + 1 >= max_steps
    )

    wrapper_root = cfg_path.parents[1]
    unity_logs = list(wrapper_root.glob("unity_import*.log"))

    return RunRecord(
        source_collection=source_collection,
        batch_or_wrapper=batch_or_wrapper,
        run_name=run_dir.name,
        run_dir=str(run_dir.relative_to(ROOT)),
        cfg_path=str(cfg_path.relative_to(ROOT)),
        metrics_path=str(metrics_path.relative_to(ROOT)) if metrics_path else "",
        completed=completed,
        max_steps=max_steps,
        latest_val_step=latest_val_step,
        antialiased=cfg.get("antialiased"),
        cap_max=_safe_int(strategy.get("cap_max")),
        min_opacity=_safe_float(strategy.get("min_opacity")),
        noise_lr=_safe_float(strategy.get("noise_lr")),
        refine_every=_safe_int(strategy.get("refine_every")),
        refine_start_iter=_safe_int(strategy.get("refine_start_iter")),
        refine_stop_iter=_safe_int(strategy.get("refine_stop_iter")),
        ssim_lambda=_safe_float(cfg.get("ssim_lambda")),
        random_bkgd=cfg.get("random_bkgd"),
        scale_reg=_safe_float(cfg.get("scale_reg")),
        opacity_reg=_safe_float(cfg.get("opacity_reg")),
        use_bilateral_grid=cfg.get("use_bilateral_grid"),
        depth_loss=cfg.get("depth_loss"),
        with_ut=cfg.get("with_ut"),
        with_eval3d=cfg.get("with_eval3d"),
        visible_adam=cfg.get("visible_adam"),
        pose_opt=cfg.get("pose_opt"),
        app_opt=cfg.get("app_opt"),
        psnr=_safe_float(metrics_payload.get("psnr")),
        ssim=_safe_float(metrics_payload.get("ssim")),
        lpips=_safe_float(metrics_payload.get("lpips")),
        num_gs=_safe_int(metrics_payload.get("num_GS")),
        unity_log_count=len(unity_logs),
    )


def _collect_train_probe_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    if not TRAIN_PROBES_ROOT.exists():
        return records

    for cfg_path in sorted(TRAIN_PROBES_ROOT.glob("*/*/cfg.yml")):
        wrapper_root = cfg_path.parents[1]
        if "mcmc" not in wrapper_root.name.lower() and "mcmc" not in cfg_path.parent.name.lower():
            continue
        records.append(
            _build_record(
                source_collection="train_probes",
                batch_or_wrapper=wrapper_root.name,
                run_dir=cfg_path.parent,
                cfg_path=cfg_path,
            )
        )
    return records


def _collect_marathon_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    if not MARATHON_ROOT.exists():
        return records

    for cfg_path in sorted(MARATHON_ROOT.glob("batch_*/*/cfg.yml")):
        batch_root = cfg_path.parents[1]
        records.append(
            _build_record(
                source_collection="mcmc_marathon",
                batch_or_wrapper=batch_root.name,
                run_dir=cfg_path.parent,
                cfg_path=cfg_path,
            )
        )
    return records


def _write_csv(records: list[RunRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_row() for record in records]
    fieldnames = list(rows[0].keys()) if rows else list(RunRecord.__annotations__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(records: list[RunRecord], output_path: Path) -> None:
    summary = {
        "total_runs": len(records),
        "train_probes_runs": sum(1 for item in records if item.source_collection == "train_probes"),
        "mcmc_marathon_runs": sum(1 for item in records if item.source_collection == "mcmc_marathon"),
        "completed_runs": sum(1 for item in records if item.completed),
        "best_lpips_run": None,
    }
    candidates = [item for item in records if item.lpips is not None]
    if candidates:
        best = min(candidates, key=lambda item: item.lpips if item.lpips is not None else float("inf"))
        summary["best_lpips_run"] = best.to_row()
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    records = _collect_train_probe_records() + _collect_marathon_records()
    records.sort(key=lambda item: (item.source_collection, item.batch_or_wrapper, item.run_name))

    csv_path = REPORTS_ROOT / "mcmc_run_inventory.csv"
    summary_path = REPORTS_ROOT / "mcmc_run_inventory.summary.json"
    _write_csv(records, csv_path)
    _write_summary(records, summary_path)

    print(f"wrote {len(records)} rows -> {csv_path}")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
