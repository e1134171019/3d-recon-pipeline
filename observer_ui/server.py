from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
AGENT_ROOT = Path(os.environ.get("AGENT_TEST_ROOT", r"D:\agent_test"))
OBSERVER_EVENTS_ROOT = ROOT / "outputs" / "observer_events"
META_ACTIVITY_LATEST = OBSERVER_EVENTS_ROOT / "latest_meta_activity.json"
META_ACTIVITY_LOG = OBSERVER_EVENTS_ROOT / "meta_activity.jsonl"
PROJECT_CATALOG = Path(__file__).resolve().parent / "project_catalog.json"


WATCHED_FILES: tuple[tuple[str, Path], ...] = (
    ("event.sfm", ROOT / "outputs" / "agent_events" / "latest_sfm_complete.json"),
    ("event.train", ROOT / "outputs" / "agent_events" / "latest_train_complete.json"),
    ("event.export", ROOT / "outputs" / "agent_events" / "latest_export_complete.json"),
    ("decision.sfm", ROOT / "outputs" / "agent_decisions" / "latest_sfm_decision.json"),
    ("decision.train", ROOT / "outputs" / "agent_decisions" / "latest_train_decision.json"),
    ("decision.export", ROOT / "outputs" / "agent_decisions" / "latest_export_decision.json"),
    ("teacher.status", ROOT / "experimental" / "scaffold_gs_probe" / "latest_teacher_loop_status.json"),
    ("learner.scaffold", AGENT_ROOT / "outputs" / "offline_learning" / "historical_plus_scaffold_report.json"),
    ("learner.baseline", AGENT_ROOT / "outputs" / "offline_learning" / "augmented_pytorch_baseline_report.json"),
    ("mcmc.inventory", ROOT / "outputs" / "reports" / "mcmc_run_inventory.summary.json"),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - surfaced in observer payload.
        return None, f"{type(exc).__name__}: {exc}"


def read_jsonl_tail(path: Path, limit: int = 40) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:  # noqa: BLE001 - surfaced in observer payload.
        return [], [f"{type(exc).__name__}: {exc}"]

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    start_index = max(0, len(lines) - limit)
    for offset, line in enumerate(lines[start_index:], start=start_index + 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {offset}: {exc}")
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            errors.append(f"line {offset}: non-object JSON record")
    return records, errors


def file_state(label: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    state: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": exists,
        "mtime": None,
        "age_seconds": None,
        "size_bytes": None,
        "error": None,
    }
    if not exists:
        state["error"] = "missing"
        return state
    try:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        state.update(
            {
                "mtime": modified.isoformat(),
                "age_seconds": round((utc_now() - modified).total_seconds(), 3),
                "size_bytes": stat.st_size,
            }
        )
    except OSError as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    return state


def latest_file(pattern: str) -> Path | None:
    matches = [path for path in ROOT.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def compact_event(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "run_id": payload.get("run_id"),
        "stage": payload.get("stage"),
        "status": payload.get("status"),
        "timestamp": payload.get("timestamp"),
        "run_root": payload.get("run_root"),
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        "artifacts": payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {},
    }


def compact_decision(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    arbiter = payload.get("arbiter")
    if isinstance(arbiter, dict):
        return {
            "decision": arbiter.get("decision") or payload.get("decision"),
            "next_action": arbiter.get("next_action") or payload.get("next_action"),
            "selected_candidate_id": arbiter.get("selected_candidate_id"),
            "requires_human_review": arbiter.get("requires_human_review"),
            "written_at": arbiter.get("written_at") or payload.get("written_at"),
            "reason": arbiter.get("reason") or payload.get("reason"),
        }
    return {
        "decision": payload.get("decision"),
        "next_action": payload.get("next_action"),
        "selected_candidate_id": payload.get("selected_candidate_id"),
        "requires_human_review": payload.get("requires_human_review"),
        "written_at": payload.get("written_at"),
        "reason": payload.get("reason"),
    }


def load_named_json(label: str) -> tuple[Any | None, dict[str, Any]]:
    path = dict(WATCHED_FILES)[label]
    data, error = read_json(path)
    state = file_state(label, path)
    if error:
        state["error"] = error
    return data, state


def build_meta_activity() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    latest, latest_error = read_json(META_ACTIVITY_LATEST)
    latest_state = file_state("observer.meta_activity.latest", META_ACTIVITY_LATEST)
    if latest_error:
        latest_state["error"] = latest_error

    events, log_errors = read_jsonl_tail(META_ACTIVITY_LOG)
    log_state = file_state("observer.meta_activity.log", META_ACTIVITY_LOG)
    if log_errors:
        log_state["error"] = "; ".join(log_errors[:3])

    meta_activity = {
        "latest": latest if isinstance(latest, dict) else None,
        "events": events,
        "errors": [error for error in [latest_error, *log_errors] if error],
        "writes_formal_runtime": False,
        "scope": "observer_only",
    }
    return meta_activity, [latest_state, log_state]


def build_snapshot() -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    for label, _path in WATCHED_FILES:
        data, state = load_named_json(label)
        loaded[label] = data
        artifacts.append(state)

    deployment_path = latest_file("outputs/experiments/**/deployment_review.json")
    bridge_path = latest_file("experimental/scaffold_gs_probe/**/*_bridge_score.json")
    for label, path in (("deployment.latest", deployment_path), ("bridge.latest", bridge_path)):
        if path is None:
            artifacts.append(
                {
                    "label": label,
                    "path": None,
                    "exists": False,
                    "mtime": None,
                    "age_seconds": None,
                    "size_bytes": None,
                    "error": "missing",
                }
            )
            loaded[label] = None
            continue
        data, error = read_json(path)
        state = file_state(label, path)
        if error:
            state["error"] = error
        artifacts.append(state)
        loaded[label] = data

    meta_activity, meta_artifacts = build_meta_activity()
    artifacts.extend(meta_artifacts)

    learner_scaffold = loaded.get("learner.scaffold")
    learner_baseline = loaded.get("learner.baseline")
    teacher_status = loaded.get("teacher.status")
    mcmc_inventory = loaded.get("mcmc.inventory")
    deployment = loaded.get("deployment.latest")

    learner_for_warning = learner_scaffold if isinstance(learner_scaffold, dict) else learner_baseline
    dataset_size = learner_for_warning.get("dataset_size") if isinstance(learner_for_warning, dict) else None
    feature_dim = learner_for_warning.get("feature_dim") if isinstance(learner_for_warning, dict) else None
    overfit_warning = (
        isinstance(dataset_size, int)
        and isinstance(feature_dim, int)
        and feature_dim > 0
        and dataset_size < feature_dim * 3
    )

    return {
        "observer": {
            "status": "ok",
            "mode": "read_only",
            "server_time": utc_now().isoformat(),
            "heartbeat": utc_now().isoformat(),
            "project_root": str(ROOT),
            "agent_root": str(AGENT_ROOT),
        },
        "formal_runtime": {
            "events": {
                "sfm": compact_event(loaded.get("event.sfm")),
                "train": compact_event(loaded.get("event.train")),
                "export": compact_event(loaded.get("event.export")),
            },
            "decisions": {
                "sfm": compact_decision(loaded.get("decision.sfm")),
                "train": compact_decision(loaded.get("decision.train")),
                "export": compact_decision(loaded.get("decision.export")),
            },
        },
        "teacher": {
            "status": teacher_status.get("status") if isinstance(teacher_status, dict) else None,
            "teacher_output": teacher_status.get("teacher_output") if isinstance(teacher_status, dict) else None,
            "merged_output": teacher_status.get("merged_output") if isinstance(teacher_status, dict) else None,
            "seed_count": teacher_status.get("seed_count") if isinstance(teacher_status, dict) else None,
            "teacher_count": teacher_status.get("teacher_count") if isinstance(teacher_status, dict) else None,
            "generated_at": teacher_status.get("generated_at") if isinstance(teacher_status, dict) else None,
        },
        "learner": {
            "scaffold": learner_scaffold if isinstance(learner_scaffold, dict) else None,
            "baseline": learner_baseline if isinstance(learner_baseline, dict) else None,
            "overfit_warning": overfit_warning,
            "warning_reason": (
                f"dataset_size {dataset_size} < 3 * feature_dim {feature_dim}"
                if overfit_warning
                else None
            ),
        },
        "mcmc": mcmc_inventory if isinstance(mcmc_inventory, dict) else None,
        "deployment_review": deployment if isinstance(deployment, dict) else None,
        "meta_activity": meta_activity,
        "artifacts": artifacts,
    }


class ObserverHandler(BaseHTTPRequestHandler):
    server_version = "ObserverUI/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name.
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"status": "ok", "mode": "read_only", "heartbeat": utc_now().isoformat()})
            return
        if parsed.path == "/api/snapshot":
            self.send_json(build_snapshot())
            return
        if parsed.path == "/api/meta-activity":
            meta_activity, artifacts = build_meta_activity()
            self.send_json({"meta_activity": meta_activity, "artifacts": artifacts})
            return
        if parsed.path == "/api/catalog":
            catalog, error = read_json(PROJECT_CATALOG)
            if error or not isinstance(catalog, dict):
                self.send_json({"error": error or "invalid catalog"}, status=500)
                return
            self.send_json(catalog)
            return
        self.send_static(parsed.path)

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: str) -> None:
        target = STATIC_ROOT / "index.html" if path in ("", "/") else STATIC_ROOT / path.lstrip("/")
        try:
            resolved = target.resolve()
            static_resolved = STATIC_ROOT.resolve()
            if static_resolved not in (resolved, *resolved.parents):
                raise ValueError("path escaped static root")
            body = resolved.read_bytes()
        except Exception:  # noqa: BLE001 - returned as 404.
            self.send_error(404, "Not found")
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature.
        print(f"[observer_ui] {self.address_string()} - {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Observer UI for the 3D recon pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true", help="Print one snapshot and exit.")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(build_snapshot(), ensure_ascii=False, indent=2))
        return 0

    server = ThreadingHTTPServer((args.host, args.port), ObserverHandler)
    print(f"[observer_ui] read-only server at http://{args.host}:{args.port}")
    print(f"[observer_ui] project_root={ROOT}")
    print(f"[observer_ui] agent_root={AGENT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[observer_ui] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
