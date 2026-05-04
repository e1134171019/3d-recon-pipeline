"""Utilities module - shared functions across train_3dgs and sfm_colmap."""

import sys
import io
import json
from pathlib import Path
from datetime import datetime


def ensure_utf8_stdout() -> None:
    """Keep CLI output UTF-8 without breaking pytest's capture streams.
    
    Call this at the module level (right after imports) to ensure all output
    from this module is properly encoded in UTF-8, with graceful fallback for
    pytest and other environments that don't expose sys.stdout.buffer.
    """
    if "pytest" in sys.modules or not hasattr(sys.stdout, "buffer"):
        return
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
    )


def read_json_robust(path: Path) -> dict:
    """Read JSON file with fallback encoding support.
    
    Tries multiple common encodings (utf-8, utf-8-sig, cp950, mbcs) to handle
    files created in different environments or with BOM markers.
    
    Args:
        path: Path to JSON file to read.
        
    Returns:
        Parsed JSON dict.
        
    Raises:
        ValueError: If JSON parsing fails with all attempted encodings.
    """
    encodings = ("utf-8", "utf-8-sig", "cp950", "mbcs")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise last_error if last_error is not None else ValueError(f"無法解析 JSON：{path}")


def generate_compact_timestamp() -> str:
    """Generate compact timestamp ID for contracts and logs.
    
    Format: YYYYMMDDhhmmss (14 digits, no separators)
    
    Examples:
        >>> ts = generate_compact_timestamp()
        >>> len(ts)
        14
        >>> ts  # doctest: +SKIP
        '20260505143022'
    
    Returns:
        Compact timestamp string suitable for IDs and file names.
    """
    return datetime.now().strftime('%Y%m%d%H%M%S')


__all__ = ["ensure_utf8_stdout", "read_json_robust", "generate_compact_timestamp"]
