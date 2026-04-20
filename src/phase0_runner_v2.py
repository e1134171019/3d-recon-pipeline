from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

from src.preprocess_phase0 import get_frame_quality_metrics

app = typer.Typer(help="Phase 0 v2：使用 ffprobe / ffmpeg 產生 candidates，再做品質篩選與 frames_1600 建立。")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "viode" / "hub.mp4"


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


def _resolve_path(value: str | None, fallback: Path | None = None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if fallback is None:
        raise ValueError("缺少必要路徑")
    return fallback.resolve()


def _clear_dir(folder: Path) -> None:
    if folder.exists():
        for child in folder.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                import shutil

                shutil.rmtree(child, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)


def _iter_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _resize_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _embed_for_similarity(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    flat = small.reshape(-1)
    norm = np.linalg.norm(flat)
    return flat if norm == 0 else flat / norm


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _probe_video(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = _run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失敗：{result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    fmt = payload.get("format", {})
    return {
        "source_video": str(video_path.resolve()),
        "width": int(video_stream.get("width", 0) or 0),
        "height": int(video_stream.get("height", 0) or 0),
        "avg_frame_rate": str(video_stream.get("avg_frame_rate", "")),
        "duration_sec": float(fmt.get("duration", 0.0) or 0.0),
        "nb_frames": int(video_stream.get("nb_frames", 0) or 0),
        "codec_name": str(video_stream.get("codec_name", "")),
        "probed_at": datetime.now().isoformat(timespec="seconds"),
    }


def _extract_candidates(
    video_path: Path,
    candidate_dir: Path,
    fps_extract: float,
    use_hwaccel: bool,
    hwaccel_backend: str,
    max_side: int,
) -> int:
    _clear_dir(candidate_dir)
    output_pattern = str((candidate_dir / "candidate_%06d.jpg").resolve())
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if use_hwaccel:
        cmd.extend(["-hwaccel", hwaccel_backend])
    vf_parts = [f"fps={fps_extract}"]
    if max_side > 0:
        vf_parts.append(f"scale={max_side}:-2:flags=lanczos")
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-vf",
            ",".join(vf_parts),
            "-q:v",
            "2",
            output_pattern,
        ]
    )
    result = _run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽幀失敗：{result.stderr.strip()}")
    return len(_iter_images(candidate_dir))


def _filter_candidates(
    candidate_dir: Path,
    clean_dir: Path,
    frames_1600_dir: Path,
    blur_threshold: float,
    brightness_low: float,
    brightness_high: float,
    dedupe_similarity_threshold: float,
    max_side: int,
    max_frames: int,
    keep_every_n: int,
) -> dict[str, Any]:
    _clear_dir(clean_dir)
    _clear_dir(frames_1600_dir)

    kept = 0
    rejected_blur = 0
    rejected_brightness = 0
    rejected_duplicate = 0
    last_embedding: np.ndarray | None = None
    candidates = _iter_images(candidate_dir)

    for idx, path in enumerate(candidates):
        if keep_every_n > 1 and (idx % keep_every_n) != 0:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue

        metrics = get_frame_quality_metrics(image)
        if metrics["laplacian_var"] < blur_threshold:
            rejected_blur += 1
            continue
        if metrics["mean_brightness"] < brightness_low or metrics["mean_brightness"] > brightness_high:
            rejected_brightness += 1
            continue

        embedding = _embed_for_similarity(image)
        similarity = _cosine_similarity(last_embedding, embedding)
        if kept > 0 and similarity >= dedupe_similarity_threshold:
            rejected_duplicate += 1
            continue

        clean_path = clean_dir / f"frame_{kept:06d}.jpg"
        frame_1600_path = frames_1600_dir / clean_path.name
        cv2.imwrite(str(clean_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        resized = _resize_to_max_side(image, max_side)
        cv2.imwrite(str(frame_1600_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1
        last_embedding = embedding

        if max_frames > 0 and kept >= max_frames:
            break

    return {
        "input_candidates": len(candidates),
        "kept_frames": kept,
        "rejected_blur": rejected_blur,
        "rejected_brightness": rejected_brightness,
        "rejected_duplicate": rejected_duplicate,
    }


@app.command()
def run(
    params_json: str = typer.Option("", help="phase0_params.json；若提供則從 JSON 讀參數"),
    source_video: str = typer.Option(str(DEFAULT_VIDEO), help="原始影片路徑"),
    candidate_dir: str = typer.Option("outputs/runs/ffmpeg_phase0_trial/phase0/candidates", help="候選幀輸出目錄"),
    clean_dir: str = typer.Option("outputs/runs/ffmpeg_phase0_trial/phase0/frames_cleaned", help="過濾後幀輸出目錄"),
    frames_1600_dir: str = typer.Option("outputs/runs/ffmpeg_phase0_trial/phase0/frames_1600", help="正式工作影像集目錄"),
    summary_path: str = typer.Option("outputs/runs/ffmpeg_phase0_trial/phase0/phase0_summary.json", help="Phase 0 summary 路徑"),
    fps_extract: float = typer.Option(12.0, help="ffmpeg 抽幀 fps"),
    max_side: int = typer.Option(1600, help="frames_1600 長邊上限"),
    blur_threshold: float = typer.Option(40.0, help="模糊閾值"),
    brightness_low: float = typer.Option(25.0, help="最低亮度"),
    brightness_high: float = typer.Option(240.0, help="最高亮度"),
    dedupe_similarity_threshold: float = typer.Option(0.985, help="近重複影格相似度閾值"),
    keep_every_n: int = typer.Option(1, help="每 N 張保留一張"),
    max_frames: int = typer.Option(0, help="最多保留幀數；0=不限"),
    use_hwaccel: bool = typer.Option(True, help="是否開啟 ffmpeg 硬體解碼"),
    hwaccel_backend: str = typer.Option("auto", help="ffmpeg hwaccel backend"),
) -> None:
    if params_json:
        payload = _read_json(Path(params_json))
        phase0_params = payload.get("phase0_params", payload)
        rec = dict(phase0_params.get("recommended_params", {}))
        source_video = rec.get("source_video", source_video)
        candidate_dir = rec.get("candidate_dir", candidate_dir)
        clean_dir = rec.get("clean_dir", clean_dir)
        frames_1600_dir = rec.get("frames_1600_dir", frames_1600_dir)
        fps_extract = float(rec.get("fps_extract", fps_extract))
        max_side = int(rec.get("max_side", max_side))
        blur_threshold = float(rec.get("blur_threshold", blur_threshold))
        brightness_low = float(rec.get("brightness_low", brightness_low))
        brightness_high = float(rec.get("brightness_high", brightness_high))
        dedupe_similarity_threshold = float(rec.get("dedupe_similarity_threshold", dedupe_similarity_threshold))
        keep_every_n = int(rec.get("keep_every_n", keep_every_n))
        max_frames = int(rec.get("max_frames", max_frames))
        use_hwaccel = bool(rec.get("use_hwaccel", use_hwaccel))
        hwaccel_backend = str(rec.get("hwaccel_backend", hwaccel_backend))

        if str(summary_path).endswith("phase0_summary.json"):
            summary_path = str((Path(clean_dir).parent / "phase0_summary.json").resolve())

    source_video_path = _resolve_path(source_video, DEFAULT_VIDEO)
    candidate_dir_path = _resolve_path(candidate_dir)
    clean_dir_path = _resolve_path(clean_dir)
    frames_1600_dir_path = _resolve_path(frames_1600_dir)
    summary_path_obj = _resolve_path(summary_path)

    if not source_video_path.exists():
        raise typer.BadParameter(f"找不到影片：{source_video_path}")

    console.print(
        Panel.fit(
            f"[bold cyan]Phase 0 v2[/]\n"
            f"video = {source_video_path}\n"
            f"fps_extract = {fps_extract}\n"
            f"use_hwaccel = {use_hwaccel} ({hwaccel_backend})\n"
            f"max_side = {max_side}",
            title="ffmpeg / ffprobe",
        )
    )

    probe = _probe_video(source_video_path)
    extracted = _extract_candidates(
        source_video_path,
        candidate_dir_path,
        fps_extract=fps_extract,
        use_hwaccel=use_hwaccel,
        hwaccel_backend=hwaccel_backend,
        max_side=max_side,
    )
    filtered = _filter_candidates(
        candidate_dir=candidate_dir_path,
        clean_dir=clean_dir_path,
        frames_1600_dir=frames_1600_dir_path,
        blur_threshold=blur_threshold,
        brightness_low=brightness_low,
        brightness_high=brightness_high,
        dedupe_similarity_threshold=dedupe_similarity_threshold,
        max_side=max_side,
        max_frames=max_frames,
        keep_every_n=keep_every_n,
    )

    summary = {
        "mode": "ffmpeg_video",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "probe": probe,
        "fps_extract": fps_extract,
        "use_hwaccel": use_hwaccel,
        "hwaccel_backend": hwaccel_backend,
        "blur_threshold": blur_threshold,
        "brightness_low": brightness_low,
        "brightness_high": brightness_high,
        "dedupe_similarity_threshold": dedupe_similarity_threshold,
        "keep_every_n": keep_every_n,
        "max_frames": max_frames,
        "max_side": max_side,
        "extracted_candidates": extracted,
        **filtered,
        "candidate_dir": str(candidate_dir_path.resolve()),
        "clean_dir": str(clean_dir_path.resolve()),
        "frames_1600_dir": str(frames_1600_dir_path.resolve()),
    }
    _write_json(summary_path_obj, summary)
    console.print(f"[green]Phase 0 v2 完成[/] {summary_path_obj}")


if __name__ == "__main__":
    app()
