from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import secrets
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / ".tmp_test_workspace"


@contextmanager
def workspace_tempdir(prefix: str = "tmp"):
    """Create a writable temp directory inside the repo workspace.

    Avoid tempfile.TemporaryDirectory() here. In this Windows environment it
    creates directories that the same Python process cannot write back into.
    """
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = TEST_TMP_ROOT / f"{prefix}{secrets.token_hex(8)}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
