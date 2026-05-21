"""Group-E Export Experiment Runner
Run view-robustness export experiments on existing 30k checkpoint.
No retraining required - tests different export filter combinations.

Usage:
    python run_viewrobust_export_pr.py [--exp E1 E2 E3] [--all] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCAFFOLD_PROBE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCAFFOLD_PROBE_ROOT.parent.parent

# Default checkpoint root: latest available run under the known scene bucket.
DEFAULT_CHECKPOINT_ROOT = (
    SCAFFOLD_PROBE_ROOT
    / "repo" / "Scaffold-GS" / "outputs"
    / "factorygaussian" / "u_base_750k_aa"
)

EXPORT_SCRIPT = SCAFFOLD_PROBE_ROOT / "export_scaffold_gs_unity_ply.py"
CSV_PATH = SCAFFOLD_PROBE_ROOT / "scaffold_viewrobust_experiments.csv"
OUTPUT_ROOT = SCAFFOLD_PROBE_ROOT / "outputs" / "viewrobust_export"


def resolve_scaffold_python() -> Path | None:
    candidates = []
    env_value = os.environ.get("SCAFFOLD_PYTHON")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            SCAFFOLD_PROBE_ROOT / ".venv_scaffold" / "Scripts" / "python.exe",
            SCAFFOLD_PROBE_ROOT / "repo" / "Scaffold-GS" / ".venv" / "Scripts" / "python.exe",
            SCAFFOLD_PROBE_ROOT / "scaffold_venv" / "Scripts" / "python.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _looks_like_scaffold_checkpoint(path: Path) -> bool:
    return any(
        (path / marker).exists()
        for marker in ("cfg_args", "results.json", "point_cloud", "chkpnt30000.pth", "chkpnt7000.pth")
    )


def resolve_default_checkpoint(root: Path = DEFAULT_CHECKPOINT_ROOT) -> Path:
    env_value = os.environ.get("SCAFFOLD_CHECKPOINT")
    if env_value:
        return Path(env_value)
    if not root.exists():
        return root
    candidates = [path for path in root.iterdir() if path.is_dir() and _looks_like_scaffold_checkpoint(path)]
    if not candidates:
        candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return root
    return max(candidates, key=lambda path: path.stat().st_mtime)


SCAFFOLD_PYTHON = resolve_scaffold_python()


def load_export_experiments(csv_path: Path) -> list[dict]:
    experiments = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["group"] == "E_export":
                experiments.append(row)
    return experiments


def run_export_experiment(exp: dict, checkpoint: Path, dry_run: bool = False) -> dict:
    exp_id = exp["exp_id"]
    min_opacity = float(exp["export_min_opacity"])
    n_cameras = int(exp["export_n_cameras"])
    visibility_n = int(exp["export_visibility_n"])
    sh_fit_degree = int(exp.get("export_sh_degree") or 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = OUTPUT_ROOT / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_out = output_dir / f"{exp_id}_{timestamp}.ply"
    report_out = output_dir / f"{exp_id}_{timestamp}_report.json"
    python_exe = SCAFFOLD_PYTHON or resolve_scaffold_python()

    cmd = [
        str(python_exe) if python_exe else "<missing-scaffold-python>",
        str(EXPORT_SCRIPT),
        "-m", str(checkpoint),
        "--iteration", "-1",
        "--camera-set", "test",
        "--camera-index", "0",
        "--avg-camera-set", "train",
        "--avg-camera-count", str(n_cameras),
        "--output", str(ply_out),
        "--report-output", str(report_out),
        "--min-opacity", str(min_opacity),
    ]

    if visibility_n > 0:
        cmd += ["--visibility-filter-n", str(visibility_n)]
    if sh_fit_degree > 0:
        cmd += ["--sh-fit-degree", str(sh_fit_degree)]

    print(f"\n{'='*60}")
    print(f"[{exp_id}] {exp['hypothesis']}")
    print(
        f"  min_opacity={min_opacity}  n_cameras={n_cameras}  "
        f"visibility_n={visibility_n}  sh_fit_degree={sh_fit_degree}"
    )
    print(f"  Output: {ply_out}")
    print(f"  CMD: {' '.join(cmd)}")

    result = {
        "exp_id": exp_id,
        "timestamp": timestamp,
        "hypothesis": exp["hypothesis"],
        "expected_effect": exp["expected_effect"],
        "priority": exp["priority"],
        "export_min_opacity": min_opacity,
        "export_n_cameras": n_cameras,
        "export_visibility_n": visibility_n,
        "export_sh_degree": sh_fit_degree,
        "ply_path": str(ply_out),
        "report_path": str(report_out),
        "status": "dry_run" if dry_run else "pending",
    }

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return result
    if python_exe is None:
        result["status"] = "environment_missing"
        result["error"] = "Scaffold-GS Python executable was not found. Set SCAFFOLD_PYTHON or create a supported venv."
        print(f"  [ENV MISSING] {result['error']}")
        return result
    if not EXPORT_SCRIPT.exists():
        result["status"] = "environment_missing"
        result["error"] = f"Export script not found: {EXPORT_SCRIPT}"
        print(f"  [ENV MISSING] {result['error']}")
        return result

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        if proc.returncode == 0:
            result["stdout_tail"] = proc.stdout[-2000:] if proc.stdout else ""
            if not ply_out.exists():
                result["status"] = "failed"
                result["error"] = f"Export returned success but PLY was not created: {ply_out}"
                print(f"  [FAIL] {result['error']}")
                return result
            result["status"] = "success"
            if report_out.exists():
                try:
                    with report_out.open(encoding="utf-8") as f:
                        result["export_report"] = json.load(f)
                except json.JSONDecodeError as exc:
                    result["report_error"] = f"invalid report JSON: {exc}"
            print(f"  [OK] Export complete: {ply_out.stat().st_size / (1024*1024):.1f} MB")
        else:
            result["status"] = "failed"
            result["stderr_tail"] = proc.stderr[-2000:] if proc.stderr else ""
            print(f"  [FAIL] returncode={proc.returncode}")
            print(proc.stderr[-500:])
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        print("  [TIMEOUT] Export exceeded 15 minutes.")
    except (FileNotFoundError, OSError) as exc:
        result["status"] = "environment_missing"
        result["error"] = str(exc)
        print(f"  [ENV MISSING] {exc}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="*", help="Run specific experiment IDs (e.g. E1 E3)")
    parser.add_argument("--all", action="store_true", help="Run all E-group experiments")
    parser.add_argument("--priority", choices=["P0", "P1", "P2"], default="P0",
                        help="Only run experiments at this priority level (default: P0)")
    parser.add_argument("--checkpoint", type=Path, default=resolve_default_checkpoint(),
                        help="Path to Scaffold-GS checkpoint directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    experiments = load_export_experiments(CSV_PATH)

    if args.exp:
        experiments = [e for e in experiments if e["exp_id"] in args.exp]
    elif not args.all:
        experiments = [e for e in experiments if e["priority"] == args.priority]

    if not experiments:
        print("No matching experiments found. Use --all or specify --exp IDs.")
        sys.exit(1)

    print(f"\nRunning {len(experiments)} export experiment(s) on checkpoint:")
    print(f"  {args.checkpoint}")

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    results = []
    for exp in experiments:
        r = run_export_experiment(exp, args.checkpoint, dry_run=args.dry_run)
        results.append(r)

    # Write summary
    summary_path = OUTPUT_ROOT / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Summary written: {summary_path}")
    success = sum(1 for r in results if r["status"] == "success")
    print(f"Results: {success}/{len(results)} succeeded")

    for r in results:
        if r["status"] == "success":
            status_tag = "OK"
        elif r["status"] == "dry_run":
            status_tag = "DRY"
        else:
            status_tag = "FAIL"
        print(f"  [{status_tag}] [{r['exp_id']}] {r['status']}")
        if r["status"] == "success" and "export_report" in r:
            rpt = r["export_report"]
            print(f"      exported_splats={rpt.get('exported_splats', '?')}  "
                  f"size_mb={rpt.get('output_size_mb', '?')}")


if __name__ == "__main__":
    main()
