from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observer_ui.record_activity import build_event_from_mapping, write_event  # noqa: E402


def _now_compact() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _write_observer_event(payload: dict[str, Any]) -> None:
    write_event(build_event_from_mapping(payload))


def _read_json(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("--event-json-file must contain a JSON object")
    return loaded


def _command_after_separator(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and emit observer-only start/end events for Vue."
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--actor", default="")
    parser.add_argument("--from-actor", default="")
    parser.add_argument("--to-actor", default="")
    parser.add_argument("--channel", default="")
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("--event-json-file", type=Path, default=None)
    parser.add_argument("--related-artifact", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = _command_after_separator(args.command)
    if not command:
        raise SystemExit("missing command after --")

    base_payload = _read_json(args.event_json_file)
    for key, value in (
        ("actor", args.actor),
        ("from_actor", args.from_actor),
        ("to_actor", args.to_actor),
        ("channel", args.channel),
        ("title", args.title),
        ("summary", args.summary),
    ):
        if value:
            base_payload[key] = value
    if args.related_artifact:
        base_payload["related_artifacts"] = args.related_artifact
    base_payload.setdefault("actor", "codex_meta_evaluator")
    base_payload.setdefault("from_actor", "Dialogue AI")
    base_payload.setdefault("to_actor", "Vue Observer")
    base_payload.setdefault("channel", "observer_event")
    base_payload.setdefault("title", "Observed command")

    start_event_id = f"observed_start_{_now_compact()}"
    _write_observer_event(
        {
            **base_payload,
            "event_id": start_event_id,
            "kind": "tool_start",
            "status": "running",
            "summary": base_payload.get("summary")
            or "Observed command started; formal runtime is not modified by the observer.",
        }
    )

    try:
        completed = subprocess.run(command, cwd=args.cwd, check=False)
    except BaseException as exc:
        _write_observer_event(
            {
                **base_payload,
                "kind": "tool_result",
                "status": "failed",
                "summary": f"Observed command failed before completion: {type(exc).__name__}: {exc}",
            }
        )
        raise

    status = "ok" if completed.returncode == 0 else "failed"
    _write_observer_event(
        {
            **base_payload,
            "kind": "tool_result",
            "status": status,
            "summary": f"Observed command finished with exit code {completed.returncode}.",
        }
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
