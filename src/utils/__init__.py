"""Utilities module - shared functions across train_3dgs and sfm_colmap."""

import sys
import io
import json
from pathlib import Path


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


__all__ = ["ensure_utf8_stdout", "read_json_robust"]
