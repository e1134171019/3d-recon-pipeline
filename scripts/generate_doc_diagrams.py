from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#f5f7fb", ec="#4a5568", fontsize=12):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, x1, y1, x2, y2, text=None, color="#4a5568"):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color=color,
    )
    ax.add_patch(arrow)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, text, ha="center", va="bottom", fontsize=10, color=color)


def base_ax(figsize=(14, 8), title: str | None = None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0.02, 0.96, title, fontsize=18, fontweight="bold", va="top")
    return fig, ax


def save(fig, name: str):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def readme_pipeline():
    fig, ax = base_ax(title="Project Mainline")
    nodes = [
        (0.03, 0.42, 0.13, 0.16, "Raw Video /\nRaw Images", "#e8f1ff"),
        (0.20, 0.42, 0.13, 0.16, "Phase 0\nFrame Filter", "#eefbf2"),
        (0.37, 0.42, 0.13, 0.16, "frames_1600", "#fff8e7"),
        (0.54, 0.42, 0.13, 0.16, "COLMAP SfM\nfeature / match /\nmapper", "#f4ecff"),
        (0.71, 0.42, 0.13, 0.16, "3DGS\nTraining", "#ffeef1"),
        (0.88, 0.42, 0.09, 0.16, "PLY /\nUnity", "#e8fbfb"),
    ]
    for x, y, w, h, text, fc in nodes:
        add_box(ax, x, y, w, h, text, fc=fc)
    for i in range(len(nodes) - 1):
        x1 = nodes[i][0] + nodes[i][2]
        y1 = nodes[i][1] + nodes[i][3] / 2
        x2 = nodes[i + 1][0]
        y2 = nodes[i + 1][1] + nodes[i + 1][3] / 2
        add_arrow(ax, x1, y1, x2, y2)
    add_box(ax, 0.70, 0.73, 0.20, 0.12, "A-route experiments\nand agent reports", fc="#fff4db", fontsize=11)
    add_arrow(ax, 0.77, 0.58, 0.79, 0.73)
    save(fig, "readme_pipeline_overview.png")


def readme_layers():
    fig, ax = base_ax(title="Hybrid Architecture")
    add_box(ax, 0.12, 0.67, 0.76, 0.18, "Agent Layer\nparams-json / reports / rerun decisions / experiment orchestration", fc="#fff1d6")
    add_box(ax, 0.12, 0.40, 0.76, 0.18, "Pipeline Layer\nsrc/*.py / runners / path contracts / validation / export", fc="#eaf5ff")
    add_box(ax, 0.12, 0.13, 0.76, 0.18, "Compute Layer\nOpenCV / COLMAP / gsplat / CUDA / Unity plugin", fc="#eef8ee")
    add_arrow(ax, 0.50, 0.67, 0.50, 0.58)
    add_arrow(ax, 0.50, 0.40, 0.50, 0.31)
    save(fig, "readme_hybrid_layers.png")


def readme_logic():
    fig, ax = base_ax(title="Current Training Decision Logic")
    add_box(ax, 0.05, 0.72, 0.22, 0.14, "Freeze\nframes_1600 + sparse/2", fc="#eaf5ff")
    add_box(ax, 0.38, 0.72, 0.22, 0.14, "Run train-only\nexperiments", fc="#eef8ee")
    add_box(ax, 0.71, 0.72, 0.18, 0.14, "Pass target?", fc="#fff1d6")
    add_box(ax, 0.66, 0.42, 0.26, 0.14, "Keep A route and move\ninto Unity / next phase", fc="#eef8ee")
    add_box(ax, 0.22, 0.42, 0.28, 0.14, "2x2 / probe follow-up\nfind dominant factors", fc="#fff8e7")
    add_box(ax, 0.18, 0.14, 0.36, 0.14, "Move upstream:\nSfM / Phase 0 / feature system", fc="#ffeef1")
    add_box(ax, 0.62, 0.14, 0.22, 0.14, "B-route evaluation\nALIKED + LightGlue", fc="#f4ecff")
    add_arrow(ax, 0.27, 0.79, 0.38, 0.79)
    add_arrow(ax, 0.60, 0.79, 0.71, 0.79)
    add_arrow(ax, 0.80, 0.72, 0.80, 0.56, "yes")
    add_arrow(ax, 0.71, 0.72, 0.40, 0.56, "no")
    add_arrow(ax, 0.36, 0.42, 0.36, 0.28)
    add_arrow(ax, 0.54, 0.21, 0.62, 0.21)
    save(fig, "readme_training_logic.png")


def map_priority_mainline():
    fig, ax = base_ax(title="Map-first Development Mainline")
    boxes = [
        (0.05, 0.42, 0.15, 0.16, "frames_1600", "#fff8e7"),
        (0.26, 0.42, 0.15, 0.16, "SfM", "#f4ecff"),
        (0.47, 0.42, 0.15, 0.16, "3DGS", "#ffeef1"),
        (0.68, 0.42, 0.15, 0.16, "Unity review", "#e8fbfb"),
        (0.84, 0.42, 0.11, 0.16, "A/B +\nagent", "#fff1d6"),
    ]
    for x, y, w, h, text, fc in boxes:
        add_box(ax, x, y, w, h, text, fc=fc)
    for i in range(len(boxes) - 1):
        add_arrow(ax, boxes[i][0] + boxes[i][2], 0.50, boxes[i + 1][0], 0.50)
    save(fig, "map_priority_mainline.png")


def map_priority_logic():
    fig, ax = base_ax(title="A/B and Optimization Logic")
    add_box(ax, 0.08, 0.72, 0.20, 0.14, "A-route baseline", fc="#eaf5ff")
    add_box(ax, 0.39, 0.72, 0.20, 0.14, "train-only\nexperiments", fc="#eef8ee")
    add_box(ax, 0.71, 0.72, 0.18, 0.14, "Reach map\nquality bar?", fc="#fff1d6")
    add_box(ax, 0.69, 0.44, 0.22, 0.14, "Close map stage\nand extend later", fc="#eef8ee")
    add_box(ax, 0.34, 0.44, 0.25, 0.14, "follow-up 2x2\nor probe tests", fc="#fff8e7")
    add_box(ax, 0.07, 0.17, 0.28, 0.14, "Move to SfM / Phase 0\nupstream tuning", fc="#ffeef1")
    add_box(ax, 0.48, 0.17, 0.32, 0.14, "B-route upstream test\nALIKED + LightGlue", fc="#f4ecff")
    add_arrow(ax, 0.28, 0.79, 0.39, 0.79)
    add_arrow(ax, 0.59, 0.79, 0.71, 0.79)
    add_arrow(ax, 0.80, 0.72, 0.80, 0.58, "yes")
    add_arrow(ax, 0.71, 0.72, 0.47, 0.58, "no")
    add_arrow(ax, 0.42, 0.44, 0.21, 0.31)
    add_arrow(ax, 0.35, 0.24, 0.48, 0.24)
    save(fig, "map_priority_ab_logic.png")


def map_priority_agent():
    fig, ax = base_ax(title="Agent Role During Map Stage")
    add_box(ax, 0.06, 0.42, 0.18, 0.16, "agent report", fc="#fff1d6")
    add_box(ax, 0.35, 0.42, 0.18, 0.16, "pass / fail\ndecision", fc="#eaf5ff")
    add_box(ax, 0.64, 0.56, 0.24, 0.14, "allow next stage\ntrain / export / compare", fc="#eef8ee")
    add_box(ax, 0.64, 0.26, 0.24, 0.14, "recovery advice\nand next rerun params", fc="#ffeef1")
    add_arrow(ax, 0.24, 0.50, 0.35, 0.50)
    add_arrow(ax, 0.53, 0.50, 0.64, 0.63, "pass")
    add_arrow(ax, 0.53, 0.50, 0.64, 0.33, "fail")
    save(fig, "map_priority_agent_logic.png")


def phase0_v2_current_vs_v2():
    fig, ax = base_ax(title="Phase 0: Current vs V2")
    add_box(ax, 0.05, 0.60, 0.38, 0.22, "Current\nraw video -> OpenCV VideoCapture -> frames_cleaned\n-> downscale -> frames_1600 -> SfM", fc="#ffeef1")
    add_box(ax, 0.57, 0.60, 0.38, 0.22, "V2\nraw video / raw photos -> ffprobe -> ffmpeg / NVDEC\n-> phase0_candidates -> Python filter -> frames_1600 -> SfM", fc="#eef8ee")
    add_box(ax, 0.15, 0.20, 0.18, 0.14, "single-script\nprototype", fc="#fff8e7")
    add_box(ax, 0.67, 0.20, 0.18, 0.14, "split layers +\nagent control", fc="#eaf5ff")
    save(fig, "phase0_v2_current_vs_v2.png")


def phase0_v2_layers():
    fig, ax = base_ax(title="Phase 0 V2 Layers")
    add_box(ax, 0.10, 0.67, 0.80, 0.18, "Agent Layer\nphase0_params.json / media probe / reports", fc="#fff1d6")
    add_box(ax, 0.10, 0.40, 0.80, 0.18, "Pipeline Layer\nprobe_media.py / extract_frames_ffmpeg.py / filter_frames_phase0.py / build_frames_1600.py / phase0_runner_v2.py", fc="#eaf5ff")
    add_box(ax, 0.10, 0.13, 0.80, 0.18, "Compute Layer\nffprobe / ffmpeg / NVDEC / OpenCV", fc="#eef8ee")
    add_arrow(ax, 0.50, 0.67, 0.50, 0.58)
    add_arrow(ax, 0.50, 0.40, 0.50, 0.31)
    save(fig, "phase0_v2_layers.png")


def phase0_v2_decision():
    fig, ax = base_ax(title="Phase 0 Decision Flow")
    add_box(ax, 0.04, 0.70, 0.18, 0.14, "raw video /\nraw photos", fc="#eaf5ff")
    add_box(ax, 0.29, 0.70, 0.18, 0.14, "media probe", fc="#eef8ee")
    add_box(ax, 0.54, 0.70, 0.18, 0.14, "video or photos?", fc="#fff1d6")
    add_box(ax, 0.80, 0.77, 0.15, 0.12, "ffmpeg /\nNVDEC", fc="#f4ecff")
    add_box(ax, 0.80, 0.58, 0.15, 0.12, "image set\nnormalize", fc="#fff8e7")
    add_box(ax, 0.54, 0.35, 0.22, 0.14, "quality filter:\nblur / brightness /\ndedupe / optional NIMA", fc="#ffeef1")
    add_box(ax, 0.28, 0.10, 0.20, 0.14, "frames_cleaned", fc="#eef8ee")
    add_box(ax, 0.58, 0.10, 0.20, 0.14, "frames_1600", fc="#fff8e7")
    add_box(ax, 0.84, 0.10, 0.10, 0.14, "SfM", fc="#e8fbfb")
    add_arrow(ax, 0.22, 0.77, 0.29, 0.77)
    add_arrow(ax, 0.47, 0.77, 0.54, 0.77)
    add_arrow(ax, 0.72, 0.77, 0.80, 0.83, "video")
    add_arrow(ax, 0.72, 0.77, 0.80, 0.64, "photos")
    add_arrow(ax, 0.87, 0.77, 0.64, 0.49)
    add_arrow(ax, 0.87, 0.58, 0.64, 0.49)
    add_arrow(ax, 0.54, 0.35, 0.38, 0.24)
    add_arrow(ax, 0.48, 0.17, 0.58, 0.17)
    add_arrow(ax, 0.78, 0.17, 0.84, 0.17)
    save(fig, "phase0_v2_decision_flow.png")


def phase0_v2_agent_tree():
    fig, ax = base_ax(title="Phase 0 Agent Decision Tree")
    add_box(ax, 0.10, 0.74, 0.22, 0.14, "phase0_media_probe.json", fc="#eaf5ff")
    add_box(ax, 0.43, 0.74, 0.22, 0.14, "video too long /\ntoo large?", fc="#fff1d6")
    add_box(ax, 0.78, 0.80, 0.16, 0.12, "change fps /\ntime window /\nuse NVDEC", fc="#eef8ee")
    add_box(ax, 0.78, 0.58, 0.16, 0.12, "quality unstable?", fc="#fff8e7")
    add_box(ax, 0.49, 0.34, 0.28, 0.14, "raise blur / brightness /\ndedupe thresholds", fc="#ffeef1")
    add_box(ax, 0.12, 0.34, 0.22, 0.14, "keep defaults", fc="#eef8ee")
    add_box(ax, 0.34, 0.10, 0.32, 0.14, "emit phase0_params.json", fc="#f4ecff")
    add_arrow(ax, 0.32, 0.81, 0.43, 0.81)
    add_arrow(ax, 0.65, 0.81, 0.78, 0.86, "yes")
    add_arrow(ax, 0.65, 0.81, 0.78, 0.64, "no")
    add_arrow(ax, 0.86, 0.58, 0.63, 0.48, "yes")
    add_arrow(ax, 0.78, 0.58, 0.34, 0.41, "no")
    add_arrow(ax, 0.49, 0.34, 0.50, 0.24)
    add_arrow(ax, 0.23, 0.34, 0.45, 0.24)
    add_arrow(ax, 0.86, 0.80, 0.58, 0.24)
    save(fig, "phase0_v2_agent_tree.png")


def main():
    ensure_dir()
    readme_pipeline()
    readme_layers()
    readme_logic()
    map_priority_mainline()
    map_priority_logic()
    map_priority_agent()
    phase0_v2_current_vs_v2()
    phase0_v2_layers()
    phase0_v2_decision()
    phase0_v2_agent_tree()
    print(FIG_DIR)


if __name__ == "__main__":
    main()
