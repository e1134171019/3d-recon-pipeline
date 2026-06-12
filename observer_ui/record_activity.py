from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVENT_ROOT = ROOT / "outputs" / "observer_events"
LATEST_PATH = EVENT_ROOT / "latest_meta_activity.json"
LOG_PATH = EVENT_ROOT / "meta_activity.jsonl"

KIND_CHOICES = (
    "task_start",
    "docs_read",
    "plan_update",
    "tool_start",
    "tool_result",
    "file_read",
    "file_edit",
    "test_run",
    "decision_review",
    "blocked",
    "final_summary",
)
STATUS_CHOICES = ("running", "ok", "warning", "failed")


def local_now() -> datetime:
    return datetime.now().astimezone()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    return value


def _loads_json_bytes(raw: bytes) -> dict[str, Any]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-16", "cp950", "mbcs"):
        try:
            loaded = json.loads(raw.decode(encoding))
            if not isinstance(loaded, dict):
                raise ValueError("JSON input must contain an object")
            return loaded
        except (UnicodeError, json.JSONDecodeError, LookupError, ValueError) as exc:
            last_error = exc
    text = raw.decode("utf-8", errors="replace")
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("JSON input must contain an object")
    return loaded


def _load_json_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.json_file:
        payload.update(_loads_json_bytes(Path(args.json_file).read_bytes()))
    if args.json_stdin:
        payload.update(_loads_json_bytes(sys.stdin.buffer.read()))
    return payload


def build_event_from_mapping(values: dict[str, Any]) -> dict[str, Any]:
    kind = values.get("kind")
    status = values.get("status")
    title = values.get("title")
    if kind not in KIND_CHOICES:
        raise ValueError(f"invalid kind: {kind!r}")
    if status not in STATUS_CHOICES:
        raise ValueError(f"invalid status: {status!r}")
    if not title:
        raise ValueError("title is required")

    timestamp = local_now().isoformat()
    compact_ts = local_now().strftime("%Y%m%d_%H%M%S")
    actor = values.get("actor") or "codex_meta_evaluator"
    return {
        "schema_version": 1,
        "event_id": values.get("event_id") or f"meta_{compact_ts}_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "actor": actor,
        "from_actor": values.get("from_actor") or actor,
        "to_actor": values.get("to_actor") or "observer_ui",
        "channel": values.get("channel") or "observer_event",
        "kind": kind,
        "status": status,
        "title": title,
        "summary": values.get("summary") or "",
        "scope": "observer_only",
        "source": "dialogue_ai",
        "related_artifacts": _as_list(
            values.get("related_artifacts", values.get("related_artifact"))
        ),
        "writes_formal_runtime": False,
    }


def build_event(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_json_payload(args)
    cli_values = {
        "kind": args.kind,
        "status": args.status,
        "title": args.title,
        "summary": args.summary,
        "actor": args.actor,
        "from_actor": args.from_actor,
        "to_actor": args.to_actor,
        "channel": args.channel,
        "event_id": args.event_id,
        "related_artifact": args.related_artifact,
    }
    for key, value in cli_values.items():
        if value not in ("", None, []):
            payload[key] = value
    return build_event_from_mapping(payload)


def write_event(payload: dict[str, Any]) -> None:
    payload = _sanitize_for_json(payload)
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
    LATEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Record observer-only dialogue AI activity.")
    parser.add_argument("--kind", choices=KIND_CHOICES)
    parser.add_argument("--status", choices=STATUS_CHOICES)
    parser.add_argument("--title", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--actor", default="codex_meta_evaluator")
    parser.add_argument("--from-actor", default="")
    parser.add_argument("--to-actor", default="")
    parser.add_argument("--channel", default="observer_event")
    parser.add_argument("--event-id", default="")
    parser.add_argument("--related-artifact", action="append", default=[])
    parser.add_argument("--json-file", default="", help="Read UTF-8 JSON object and merge CLI overrides.")
    parser.add_argument("--json-stdin", action="store_true", help="Read UTF-8 JSON object from stdin.")
    args = parser.parse_args()

    payload = build_event(args)
    write_event(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
