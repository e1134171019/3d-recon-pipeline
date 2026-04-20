#!/usr/bin/env python
"""Clean agent mailbox snapshots while keeping shared latest state."""

from __future__ import annotations

import argparse
from pathlib import Path


MAILBOX_DIRS = ("agent_events", "agent_decisions")
KEEP_PREFIX = "latest_"


def collect_stale_json(mailbox_root: Path) -> list[Path]:
    """Return JSON snapshots that are safe to remove."""
    stale: list[Path] = []
    if not mailbox_root.exists():
        return stale

    for path in sorted(mailbox_root.glob("*.json")):
        if path.name.startswith(KEEP_PREFIX):
            continue
        stale.append(path)
    return stale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean timestamped JSON snapshots in outputs/agent_events and "
            "outputs/agent_decisions while keeping latest_* shared state."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Outputs root that contains agent_events and agent_decisions.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Default is dry-run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs_root = args.outputs_root.resolve()

    stale_files: list[Path] = []
    for mailbox_name in MAILBOX_DIRS:
        stale_files.extend(collect_stale_json(outputs_root / mailbox_name))

    if not stale_files:
        print(f"[agent-cleanup] nothing to remove under {outputs_root}")
        return 0

    mode = "apply" if args.apply else "dry-run"
    print(f"[agent-cleanup] mode={mode}")
    for path in stale_files:
        print(f"[agent-cleanup] stale={path}")

    if args.apply:
        for path in stale_files:
            path.unlink()
        print(f"[agent-cleanup] removed={len(stale_files)}")
    else:
        print(f"[agent-cleanup] pending={len(stale_files)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
