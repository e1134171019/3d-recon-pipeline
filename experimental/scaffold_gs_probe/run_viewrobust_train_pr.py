"""Group-T Training Experiment Runner
Run view-robustness training probes on the gsplat Scaffold sandbox.

This runner intentionally stays in the sandbox and only supports knobs that
`gsplat_scaffold/examples/simple_trainer_scaffold.py` currently exposes.
Experiments requiring code changes (for example distance-conditioned opacity /
covariance) are reported as unsupported instead of being silently ignored.

Usage:
    python run_viewrobust_train_pr.py [--exp T1 T3] [--all] [--dry-run]
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
CSV_PATH = SCAFFOLD_PROBE_ROOT / "scaffold_viewrobust_experiments.csv"
OUTPUT_ROOT = SCAFFOLD_PROBE_ROOT / "outputs" / "viewrobust_train"
GSPLAT_ROOT = SCAFFOLD_PROBE_ROOT / "gsplat_scaffold"
EXAMPLES_ROOT = GSPLAT_ROOT / "examples"
TRAINER = EXAMPLES_ROOT / "simple_trainer_scaffold.py"
SCAFFOLD_PYTHON = SCAFFOLD_PROBE_ROOT / "scaffold_venv" / "Scripts" / "python.exe"
DEFAULT_DATA_DIR = Path(
    r"C:\3d-recon-pipeline\experimental\scaffold_gs_probe\data"
    r"\factorygaussian\u_base_750k_aa"
)


def load_train_experiments(csv_path: Path) -> list[dict]:
    experiments = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["group"] == "T_train":
                experiments.append(row)
    return experiments


def build_training_command(exp: dict, data_dir: Path, result_dir: Path) -> list[str]:
    if exp["add_opacity_dist"] == "True":
        raise NotImplementedError("add_opacity_dist is not exposed by simple_trainer_scaffold.py yet")
    if exp["add_cov_dist"] == "True":
        raise NotImplementedError("add_cov_dist is not exposed by simple_trainer_scaffold.py yet")
    if exp["add_color_dist"] == "True":
        raise NotImplementedError("add_color_dist is not exposed by simple_trainer_scaffold.py yet")

    iterations = int(exp["iterations"])
    cmd = [
        str(SCAFFOLD_PYTHON),
        str(TRAINER),
        "--data-dir", str(data_dir),
        "--data-factor", "8",
        "--result-dir", str(result_dir),
        "--max-steps", str(iterations),
        "--eval-steps", str(iterations),
        "--save-steps", str(iterations),
        "--disable-viewer",
        "--voxel-size", str(float(exp["voxel_size"])),
        "--ssim-lambda", "0.3",
        "--feat-dim", str(int(exp["feat_dim"])),
        "--n-feat-offsets", str(int(exp["n_offsets"])),
        "--strategy.prune-opa", str(float(exp["prune_opa"])),
    ]
    if exp.get("train_antialiased") == "True":
        cmd.append("--antialiased")
    if exp.get("train_random_bkgd") == "True":
        cmd.append("--random-bkgd")
    return cmd


def build_training_env() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = [str(GSPLAT_ROOT), str(EXAMPLES_ROOT)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))
    return env


def run_train_experiment(exp: dict, data_dir: Path, dry_run: bool = False) -> dict:
    exp_id = exp["exp_id"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / exp_id / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "exp_id": exp_id,
        "timestamp": timestamp,
        "hypothesis": exp["hypothesis"],
        "expected_effect": exp["expected_effect"],
        "priority": exp["priority"],
        "result_dir": str(output_dir),
        "status": "dry_run" if dry_run else "pending",
    }

    try:
        cmd = build_training_command(exp, data_dir, output_dir)
    except NotImplementedError as exc:
        result["status"] = "unsupported"
        result["reason"] = str(exc)
        print(f"\n[{exp_id}] unsupported: {exc}")
        return result

    print(f"\n{'=' * 60}")
    print(f"[{exp_id}] {exp['hypothesis']}")
    print(f"  Output: {output_dir}")
    print(f"  CWD: {GSPLAT_ROOT}")
    print(f"  CMD: {' '.join(cmd)}")

    result["cmd"] = cmd
    result["cwd"] = str(GSPLAT_ROOT)
    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return result

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            timeout=7200,
            cwd=GSPLAT_ROOT,
            env=build_training_env(),
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["status"] = "success"
            print("  [OK] Training probe complete.")
        else:
            result["status"] = "failed"
            print(f"  [FAIL] returncode={proc.returncode}")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        print("  [TIMEOUT] Training exceeded 2 hours.")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="*", help="Run specific experiment IDs (e.g. T1 T9)")
    parser.add_argument("--all", action="store_true", help="Run all T-group experiments")
    parser.add_argument(
        "--priority",
        choices=["P1", "P2"],
        default="P1",
        help="Only run experiments at this priority level (default: P1)",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    if not SCAFFOLD_PYTHON.exists():
        print(f"ERROR: scaffold python not found: {SCAFFOLD_PYTHON}")
        sys.exit(1)
    if not TRAINER.exists():
        print(f"ERROR: trainer not found: {TRAINER}")
        sys.exit(1)
    if not GSPLAT_ROOT.exists():
        print(f"ERROR: gsplat root not found: {GSPLAT_ROOT}")
        sys.exit(1)
    if not args.data_dir.exists():
        print(f"ERROR: data dir not found: {args.data_dir}")
        sys.exit(1)

    experiments = load_train_experiments(CSV_PATH)
    if args.exp:
        experiments = [e for e in experiments if e["exp_id"] in args.exp]
    elif not args.all:
        experiments = [e for e in experiments if e["priority"] == args.priority]

    if not experiments:
        print("No matching experiments found. Use --all or specify --exp IDs.")
        sys.exit(1)

    print(f"\nRunning {len(experiments)} training experiment(s) on data dir:")
    print(f"  {args.data_dir}")

    results = []
    for exp in experiments:
        results.append(run_train_experiment(exp, args.data_dir, dry_run=args.dry_run))

    summary_path = OUTPUT_ROOT / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Summary written: {summary_path}")
    for result in results:
        if result["status"] == "success":
            status_tag = "OK"
        elif result["status"] == "unsupported":
            status_tag = "UNSUPPORTED"
        elif result["status"] == "dry_run":
            status_tag = "DRY"
        else:
            status_tag = "FAIL"
        print(f"  [{status_tag}] [{result['exp_id']}] {result['status']}")


if __name__ == "__main__":
    main()
