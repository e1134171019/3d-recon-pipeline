from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


app = typer.Typer(help="上游 2x2 實驗矩陣 runner（Phase 0 / SfM / frozen train baseline）")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = PROJECT_ROOT / "outputs" / "experiments" / "upstream_2x2_matrix.json"

PHASE0_KEYS = {
    "source_video",
    "source_images",
    "candidate_dir",
    "clean_dir",
    "frames_1600_dir",
    "fps_extract",
    "max_frames",
    "max_side",
    "blur_threshold",
    "brightness_low",
    "brightness_high",
    "dedupe_similarity_threshold",
    "keep_every_n",
    "use_hwaccel",
    "hwaccel_backend",
    "curation_mode",
}

SFM_KEYS = {
    "imgdir",
    "work",
    "mapper_type",
    "max_features",
    "seq_overlap",
    "loop_detection",
    "matcher",
    "sift_peak_threshold",
    "sift_edge_threshold",
    "glomap_bin",
    "validation_report",
}

TRAIN_KEYS = {
    "imgdir",
    "colmap",
    "outdir",
    "iterations",
    "sh_degree",
    "densify_until",
    "scene_scale",
    "scale_json",
    "eval_steps",
    "data_factor",
    "absgrad",
    "grow_grad2d",
    "antialiased",
    "random_bkgd",
    "validation_report",
}


def _read_json(path: Path) -> dict[str, Any]:
    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp950", "mbcs")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise last_error if last_error is not None else ValueError(f"無法解析 JSON：{path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_path(value: str | None) -> str:
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _load_matrix(matrix: Path) -> tuple[Path, dict[str, Any]]:
    matrix_path = matrix if matrix.is_absolute() else (PROJECT_ROOT / matrix)
    if not matrix_path.exists():
        raise typer.BadParameter(f"找不到矩陣檔：{matrix_path}")
    payload = _read_json(matrix_path)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise typer.BadParameter(f"矩陣格式不正確：{matrix_path}")
    return matrix_path, payload


def _matrix_name(matrix_path: Path, payload: dict[str, Any]) -> str:
    return str(payload.get("matrix_name") or matrix_path.stem)


def _matrix_root(matrix_path: Path, payload: dict[str, Any]) -> Path:
    shared = payload.get("shared", {})
    outdir_root = shared.get("outdir_root")
    if outdir_root:
        root = Path(str(outdir_root))
        return root if root.is_absolute() else (PROJECT_ROOT / root)
    return matrix_path.with_suffix("")


def _build_plan(
    matrix_path: Path,
    payload: dict[str, Any],
    experiment: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    shared = dict(payload.get("shared", {}))
    phase0_base = dict(shared.get("phase0", {}))
    sfm_base = dict(shared.get("sfm", {}))
    train_base = dict(shared.get("train", {}))

    overrides = dict(experiment.get("overrides", {}))
    phase0_overrides = dict(overrides.get("phase0", {}))
    sfm_overrides = dict(overrides.get("sfm", {}))
    train_overrides = dict(overrides.get("train", {}))

    phase0_params = {**phase0_base, **phase0_overrides}
    sfm_params = {**sfm_base, **sfm_overrides}
    train_params = {**train_base, **train_overrides}

    experiment_root = _matrix_root(matrix_path, payload) / str(experiment["name"])

    if "candidate_dir" in phase0_params:
        phase0_params["candidate_dir"] = str((experiment_root / "phase0" / "candidates").resolve())
    if "clean_dir" in phase0_params:
        phase0_params["clean_dir"] = str((experiment_root / "phase0" / "frames_cleaned").resolve())
    if "frames_1600_dir" in phase0_params:
        phase0_params["frames_1600_dir"] = str((experiment_root / "phase0" / "frames_1600").resolve())

    sfm_params["imgdir"] = phase0_params.get("frames_1600_dir", _resolve_path(sfm_params.get("imgdir")))
    sfm_params["work"] = str((experiment_root / "SfM_models" / "sift").resolve())
    sfm_params["validation_report"] = str((experiment_root / "reports" / "pointcloud_validation_report.json").resolve())

    train_params["imgdir"] = phase0_params.get("frames_1600_dir", _resolve_path(train_params.get("imgdir")))
    train_params["colmap"] = str((experiment_root / "SfM_models" / "sift" / "sparse" / "best").resolve())
    train_params["outdir"] = str((experiment_root / "3DGS_models").resolve())
    train_params["validation_report"] = str((experiment_root / "reports" / "pointcloud_validation_report.json").resolve())

    for key in ("source_video", "source_images"):
        phase0_params[key] = _resolve_path(phase0_params.get(key))
    for key in ("imgdir", "validation_report"):
        sfm_params[key] = _resolve_path(sfm_params.get(key))
    for key in ("imgdir", "colmap", "validation_report", "scale_json"):
        train_params[key] = _resolve_path(train_params.get(key))

    plan = {
        "phase0_params": {
            "profile_name": experiment["name"],
            "matrix_name": _matrix_name(matrix_path, payload),
            "question": experiment.get("question", ""),
            "comparison_role": experiment.get("comparison_role", ""),
            "hypothesis": experiment.get("hypothesis", ""),
            "recommended_params": {key: phase0_params[key] for key in PHASE0_KEYS if key in phase0_params},
        },
        "sfm_params": {
            "profile_name": experiment["name"],
            "matrix_name": _matrix_name(matrix_path, payload),
            "question": experiment.get("question", ""),
            "comparison_role": experiment.get("comparison_role", ""),
            "hypothesis": experiment.get("hypothesis", ""),
            "recommended_params": {key: sfm_params[key] for key in SFM_KEYS if key in sfm_params},
        },
        "train_params": {
            "profile_name": experiment["name"],
            "matrix_name": _matrix_name(matrix_path, payload),
            "question": experiment.get("question", ""),
            "comparison_role": experiment.get("comparison_role", ""),
            "hypothesis": experiment.get("hypothesis", ""),
            "recommended_params": {key: train_params[key] for key in TRAIN_KEYS if key in train_params},
        },
    }
    return plan, experiment_root


def _materialize_all(matrix_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    root = _matrix_root(matrix_path, payload)
    root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []

    for experiment in payload.get("experiments", []):
        plan, experiment_root = _build_plan(matrix_path, payload, experiment)
        phase0_path = experiment_root / "phase0_params.json"
        sfm_path = experiment_root / "sfm_params.json"
        train_path = experiment_root / "train_params.json"
        meta_path = experiment_root / "experiment_meta.json"
        observation_path = experiment_root / "observation_template.json"

        _write_json(phase0_path, {"phase0_params": plan["phase0_params"]})
        _write_json(sfm_path, {"sfm_params": plan["sfm_params"]})
        _write_json(train_path, {"train_params": plan["train_params"]})
        _write_json(
            meta_path,
            {
                "matrix_name": _matrix_name(matrix_path, payload),
                "experiment_name": experiment["name"],
                "question": experiment.get("question", ""),
                "comparison_role": experiment.get("comparison_role", ""),
                "hypothesis": experiment.get("hypothesis", ""),
            },
        )
        _write_json(
            observation_path,
            {
                "experiment_name": experiment["name"],
                "phase0_subjective": "",
                "sfm_subjective": "",
                "unity_subjective": "",
                "notes": "",
            },
        )

        generated.append(
            {
                "name": experiment["name"],
                "root": str(experiment_root.resolve()),
                "phase0_params_json": str(phase0_path.resolve()),
                "sfm_params_json": str(sfm_path.resolve()),
                "train_params_json": str(train_path.resolve()),
            }
        )

    _write_json(
        root / "manifest.json",
        {
            "matrix_name": _matrix_name(matrix_path, payload),
            "matrix_path": str(matrix_path.resolve()),
            "materialized_at": datetime.now().isoformat(timespec="seconds"),
            "experiments": generated,
        },
    )
    return generated


@app.command(name="list")
def list_matrix(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="上游實驗矩陣 JSON"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    table = Table(title=f"上游 2x2 矩陣：{_matrix_name(matrix_path, payload)}")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Question", style="white")
    table.add_column("Phase0", style="magenta")
    table.add_column("SfM", style="yellow")

    for experiment in payload.get("experiments", []):
        ov = experiment.get("overrides", {})
        p0 = ov.get("phase0", {})
        sfm = ov.get("sfm", {})
        phase0_summary = (
            f"curation={p0.get('curation_mode', 'base')} | "
            f"fps={p0.get('fps_extract', '-') } | "
            f"dedupe={p0.get('dedupe_similarity_threshold', '-')}"
        )
        sfm_summary = (
            f"max_features={sfm.get('max_features', 'base')} | "
            f"seq_overlap={sfm.get('seq_overlap', 'base')} | "
            f"peak={sfm.get('sift_peak_threshold', 'base')}"
        )
        table.add_row(
            str(experiment.get("name", "")),
            str(experiment.get("comparison_role", "")),
            str(experiment.get("question", "")),
            phase0_summary,
            sfm_summary,
        )
    console.print(table)


@app.command()
def materialize(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="上游實驗矩陣 JSON"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    generated = _materialize_all(matrix_path, payload)
    lines = [f"{item['name']}: {item['root']}" for item in generated]
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold green]已生成上游 2x2 參數骨架[/]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
